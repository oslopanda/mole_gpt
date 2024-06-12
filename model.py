import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np

PAD_token = 0
SOS_token = 1
EOS_token = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromSmiles(lang, smiles):
    return [lang.char2index[char] for char in smiles] + [EOS_token]

def tensorFromSmiles(lang, smiles):
    indexes = indexesFromSmiles(lang, smiles)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def pad_seq(seq, max_length):
    seq += [PAD_token for i in range(max_length - len(seq))]
    return seq

def sequence_mask(sequence_length, max_len=None):
    """
    :param sequence_length: A tensor containing the lengths of sequences in a batch. The tensor should have shape (batch_size,).
    :param max_len: The maximum length of sequences in the batch. If not provided, it is computed as the maximum value in sequence_length tensor.
    :return: A binary mask tensor of shape (batch_size, max_len), where each element is True if the corresponding position in the sequence is within the length of that sequence, and False otherwise.
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long() #note arange is [) unlike torch.range which is []
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def masked_cross_entropy(logits, target, length):
    logits_flat = logits.view(-1, logits.size(-1))  # Flatten logits
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    #target = target.unsqueeze(-1)
    #print (target.shape)
    target_flat = target.reshape(-1, 1)
    #target_flat = target.view(-1, 1)  # Flatten target

    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)  # Compute flat losses
    losses = losses_flat.view(*target.size())  # Reshape losses to match target shape for masking

    # Create a mask for each sequence in the batch
    mask = sequence_mask(sequence_length=length, max_len=logits.size(0))
    mask_flat = mask.view(-1)  # Flatten the mask to match losses_flat

    # Apply mask to flat losses
    masked_losses = losses_flat.squeeze() * mask_flat.float()  # Ensure losses_flat is squeezed to remove the redundant dimension
    masked_loss_sum = masked_losses.sum()

    # Normalize loss by the number of unmasked (non-padded) elements
    loss = masked_loss_sum / mask_flat.float().sum()

    return loss

def get_positional_encoding(max_seq_length, d_model):
    positional_enc = np.array([
        [pos / np.power(10000, (2 * (j // 2)) / d_model) for j in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_seq_length)])
    positional_enc[1:, 0::2] = np.sin(positional_enc[1:, 0::2])  # Apply sin to even indices in the array
    positional_enc[1:, 1::2] = np.cos(positional_enc[1:, 1::2])  # Apply cos to odd indices
    return torch.from_numpy(positional_enc).type(torch.FloatTensor).to(device)


class PositionalEncodingLayer(torch.nn.Module):
    def __init__(self, max_seq_length, d_model):
        super(PositionalEncodingLayer, self).__init__()
        self.pos_encoding = get_positional_encoding(max_seq_length, d_model).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_encoding[:, :x.size(1), :]

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask, key_padding_mask=None):
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=key_padding_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        # Feed-forward
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm2(tgt)
        return tgt

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class ConditionalTransformerDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, condition_vector_size, num_heads, num_layers, max_seq_length,
                 dropout=0.1):
        super(ConditionalTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        # A feed-forward network to embed the condition vector
        self.condition_embedding_net = nn.Sequential(
            nn.Linear(condition_vector_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.positional_encoding = PositionalEncodingLayer(max_seq_length, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths, condition_vector):
        # Generate autoregressive mask
        tgt_mask = generate_square_subsequent_mask(input_seq.size(0)).to(input_seq.device)
        # max_len = input_seq.size(0)
        # batch_size = input_seq.size(1)
        # key_padding_mask = torch.arange(max_len, device=input_seq.device).expand(batch_size, max_len) >= seq_lengths.unsqueeze(1)
        condition_vector = condition_vector.float()
        condition_emb = self.condition_embedding_net(condition_vector)  # Embed the condition vector
        embedded = self.embedding(input_seq) + condition_emb.unsqueeze(
            0)  # Combine condition embedding with input embedding
        embedded = self.positional_encoding(embedded)
        # Decoder layers
        output = embedded
        # print (output)
        for layer in self.layers:
            output = layer(output, tgt_mask)

        # Generate final output predictions
        # print (output)
        output = self.out(output)
        return F.log_softmax(output, dim=-1)