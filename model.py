"""
GPT4mole: Conditional Molecular Generation using Transformer Architecture

This module implements a conditional transformer decoder for generating SMILES strings
of chemical molecules based on specified molecular properties.

Author: Wen Xing
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Special tokens for sequence processing
PAD_token = 0  # Padding token for variable-length sequences
SOS_token = 1  # Start-of-sequence token
EOS_token = 2  # End-of-sequence token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def indexesFromSmiles(lang, smiles):
    """
    Convert a SMILES string to a list of token indices.

    Args:
        lang (Lang): Language object containing character-to-index mapping
        smiles (str): SMILES string representation of a molecule

    Returns:
        list: List of token indices with EOS token appended
    """
    return [lang.char2index[char] for char in smiles] + [EOS_token]


def sequence_mask(sequence_length, max_len=None):
    """
    Create a binary mask for variable-length sequences in a batch.

    This function generates a mask that indicates which positions in each sequence
    are valid (not padding). Used for masking loss computation and attention.

    Args:
        sequence_length (torch.Tensor): Tensor of shape (batch_size,) containing
                                      the actual length of each sequence
        max_len (int, optional): Maximum sequence length. If None, computed as
                               the maximum value in sequence_length

    Returns:
        torch.Tensor: Binary mask of shape (batch_size, max_len) where True
                     indicates valid positions and False indicates padding
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    # note arange is [) unlike torch.range which is []
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    """
    Compute masked cross-entropy loss for variable-length sequences.

    This function calculates cross-entropy loss while ignoring padded positions
    in the sequences. Essential for training on batches with variable-length sequences.

    Args:
        logits (torch.Tensor): Model predictions of shape (seq_len, batch_size, vocab_size)
        target (torch.Tensor): Target token indices of shape (seq_len, batch_size)
        length (torch.Tensor): Actual sequence lengths of shape (batch_size,)

    Returns:
        torch.Tensor: Scalar loss value normalized by number of non-padded tokens
    """
    logits_flat = logits.view(-1, logits.size(-1))  # Flatten logits
    log_probs_flat = F.log_softmax(logits_flat, dim=-1)
    # target = target.unsqueeze(-1)
    # print (target.shape)
    target_flat = target.reshape(-1, 1)
    # target_flat = target.view(-1, 1)  # Flatten target, for some reason view not work needs reshape

    # Compute flat losses
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # Reshape losses to match target shape for masking
    losses = losses_flat.view(*target.size())

    # Create a mask for each sequence in the batch
    mask = sequence_mask(sequence_length=length, max_len=logits.size(0))
    mask_flat = mask.view(-1)  # Flatten the mask to match losses_flat

    # Apply mask to flat losses
    # Ensure losses_flat is squeezed to remove the redundant dimension
    masked_losses = losses_flat.squeeze() * mask_flat.float()
    masked_loss_sum = masked_losses.sum()

    # Normalize loss by the number of unmasked (non-padded) elements
    loss = masked_loss_sum / mask_flat.float().sum()

    return loss


def get_positional_encoding(max_seq_length, d_model):
    """
    Generate sinusoidal positional encodings for transformer models.

    Creates position-dependent encodings that help the model understand
    the order of tokens in a sequence, following the original Transformer paper.

    Args:
        max_seq_length (int): Maximum sequence length to generate encodings for
        d_model (int): Model dimension (hidden size)

    Returns:
        torch.Tensor: Positional encodings of shape (max_seq_length, d_model)
    """
    positional_enc = np.array([
        [pos / np.power(10000, (2 * (j // 2)) / d_model)
         for j in range(d_model)]
        if pos != 0 else np.zeros(d_model) for pos in range(max_seq_length)])
    # Apply sin to even indices in the array
    positional_enc[1:, 0::2] = np.sin(positional_enc[1:, 0::2])
    positional_enc[1:, 1::2] = np.cos(
        positional_enc[1:, 1::2])  # Apply cos to odd indices
    return torch.from_numpy(positional_enc).type(torch.FloatTensor).to(device)


class PositionalEncodingLayer(torch.nn.Module):
    """
    Learnable positional encoding layer for transformer models.

    This layer adds positional information to input embeddings, allowing
    the model to understand token positions in sequences.
    """

    def __init__(self, max_seq_length, d_model):
        """
        Initialize the positional encoding layer.

        Args:
            max_seq_length (int): Maximum sequence length
            d_model (int): Model dimension (hidden size)
        """
        super(PositionalEncodingLayer, self).__init__()
        self.pos_encoding = get_positional_encoding(
            max_seq_length, d_model).unsqueeze(0)

    def forward(self, x):
        """
        Add positional encodings to input embeddings.

        Args:
            x (torch.Tensor): Input embeddings of shape (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: Input embeddings with positional encodings added
        """
        return x + self.pos_encoding[:, :x.size(1), :]


class TransformerDecoderLayer(nn.Module):
    """
    Single transformer decoder layer with self-attention and feed-forward networks.

    Implements the standard transformer decoder layer architecture with:
    - Masked self-attention mechanism
    - Position-wise feed-forward network
    - Residual connections and layer normalization
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        """
        Initialize transformer decoder layer.

        Args:
            hidden_size (int): Hidden dimension of the model
            num_heads (int): Number of attention heads
            dropout (float): Dropout probability
        """
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask, key_padding_mask=None):
        """
        Forward pass through the transformer decoder layer.

        Args:
            tgt (torch.Tensor): Target sequence of shape (seq_len, batch_size, hidden_size)
            tgt_mask (torch.Tensor): Causal mask for autoregressive generation
            key_padding_mask (torch.Tensor, optional): Padding mask

        Returns:
            torch.Tensor: Output of shape (seq_len, batch_size, hidden_size)
        """
        attn_output, _ = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=key_padding_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        # Feed-forward
        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm2(tgt)
        return tgt


def generate_square_subsequent_mask(sz):
    """
    Generate a square causal mask for autoregressive generation.

    Creates a mask that prevents the model from attending to future tokens,
    ensuring autoregressive (left-to-right) generation.

    Args:
        sz (int): Size of the square mask (sequence length)

    Returns:
        torch.Tensor: Causal mask of shape (sz, sz) with -inf for masked positions
    """
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float(
        '-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class ConditionalTransformerDecoder(nn.Module):
    """
    Conditional Transformer Decoder for molecular generation.

    This model generates SMILES strings conditioned on molecular properties.
    It uses a standard transformer decoder architecture with the addition of
    condition vector embedding that influences the generation process.

    The model works by:
    1. Embedding input tokens and condition vectors
    2. Adding positional encodings
    3. Processing through multiple transformer decoder layers
    4. Outputting probability distributions over the vocabulary
    """

    def __init__(self, output_size, hidden_size, condition_vector_size, num_heads, num_layers, max_seq_length,
                 dropout=0.1):
        """
        Initialize the conditional transformer decoder.

        Args:
            output_size (int): Size of the output vocabulary (number of unique tokens)
            hidden_size (int): Hidden dimension of the model
            condition_vector_size (int): Dimension of the condition vector (molecular properties)
            num_heads (int): Number of attention heads in each transformer layer
            num_layers (int): Number of transformer decoder layers
            max_seq_length (int): Maximum sequence length for positional encoding
            dropout (float): Dropout probability for regularization
        """
        super(ConditionalTransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        # A feed-forward network to embed the condition vector
        self.condition_embedding_net = nn.Sequential(
            nn.Linear(condition_vector_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.positional_encoding = PositionalEncodingLayer(
            max_seq_length, hidden_size)
        self.layers = nn.ModuleList(
            [TransformerDecoderLayer(hidden_size, num_heads, dropout) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, seq_lengths, condition_vector):
        """
        Forward pass through the conditional transformer decoder.

        Args:
            input_seq (torch.Tensor): Input token sequence of shape (seq_len, batch_size)
            seq_lengths (torch.Tensor): Actual sequence lengths (currently unused but kept for compatibility)
            condition_vector (torch.Tensor): Condition vector of shape (batch_size, condition_vector_size)
                                           containing molecular properties

        Returns:
            torch.Tensor: Log probabilities over vocabulary of shape (seq_len, batch_size, vocab_size)
        """
        # Generate autoregressive mask to prevent attending to future tokens
        tgt_mask = generate_square_subsequent_mask(
            input_seq.size(0)).to(input_seq.device)
        # max_len = input_seq.size(0)
        # batch_size = input_seq.size(1)
        # key_padding_mask = torch.arange(max_len, device=input_seq.device).expand(batch_size, max_len) >= seq_lengths.unsqueeze(1)
        condition_vector = condition_vector.float()
        condition_emb = self.condition_embedding_net(
            condition_vector)  # Embed the condition vector
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
