from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

PAD_token = 0
SOS_token = 1
EOS_token = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class SMILESDataset(Dataset):
    def __init__(self, pairs, conditions, input_lang, output_lang):
        self.pairs = pairs
        self.conditions = conditions
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        input_tensor = self.tensorFromSMILES(self.input_lang, self.pairs[idx][0])
        target_tensor = self.tensorFromSMILES(self.output_lang, self.pairs[idx][1])
        condition_tensor = self.conditions[idx]
        return input_tensor, target_tensor, condition_tensor

    @staticmethod
    def tensorFromSMILES(lang, smiles):
        indexes = [lang.char2index[char] for char in smiles]
        indexes.append(EOS_token)  # Append EOS token
        return torch.tensor(indexes, dtype=torch.long)  # Shape: [seq_len]


def collate_fn(batch):
    input_tensors, target_tensors, condition_tensors = zip(*batch)

    # Calculate lengths before padding
    input_lengths = [len(tensor) for tensor in input_tensors]
    target_lengths = [len(tensor) for tensor in target_tensors]

    # Move tensors to the correct device and add SOS_token for input sequences
    input_tensors = [torch.cat([torch.tensor([SOS_token], dtype=torch.long, device=device),
                                tensor[:-1].to(device)], dim=0) for tensor in input_tensors]

    # Pad sequences
    input_tensors = torch.nn.utils.rnn.pad_sequence(input_tensors, padding_value=PAD_token, batch_first=True)
    target_tensors = torch.nn.utils.rnn.pad_sequence(target_tensors, padding_value=PAD_token, batch_first=True)

    # Convert lengths to tensor
    input_lengths = torch.tensor(input_lengths, dtype=torch.long, device=device)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long, device=device)
    condition_tensors = torch.tensor(np.array(condition_tensors), dtype=torch.float, device=device)

    return input_tensors.to(device), input_lengths.to(device), target_tensors.to(device), target_lengths.to(
        device), condition_tensors.to(device)