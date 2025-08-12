"""
GPT4mole: Dataset Classes for Molecular Data Processing

This module contains PyTorch Dataset classes and utilities for handling
molecular SMILES data with conditional information for training.

Author: Wen Xing
License: MIT
"""

from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

# Special tokens for sequence processing
PAD_token = 0  # Padding token for variable-length sequences
SOS_token = 1  # Start-of-sequence token
EOS_token = 2  # End-of-sequence token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SMILESDataset(Dataset):
    """
    PyTorch Dataset for SMILES strings with conditional information.

    This dataset handles molecular SMILES strings paired with condition vectors
    (molecular properties) for training conditional molecular generation models.

    Attributes:
        pairs (list): List of (input_smiles, target_smiles) tuples
        conditions (list): List of condition vectors (molecular properties)
        input_lang (Lang): Language object for input tokenization
        output_lang (Lang): Language object for output tokenization
    """

    def __init__(self, pairs, conditions, input_lang, output_lang):
        """
        Initialize the SMILES dataset.

        Args:
            pairs (list): List of (input_smiles, target_smiles) tuples
            conditions (list): List of condition vectors for each SMILES
            input_lang (Lang): Language object for input tokenization
            output_lang (Lang): Language object for output tokenization
        """
        self.pairs = pairs
        self.conditions = conditions
        self.input_lang = input_lang
        self.output_lang = output_lang

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple: (input_tensor, target_tensor, condition_tensor)
        """
        input_tensor = self.tensorFromSMILES(
            self.input_lang, self.pairs[idx][0])
        target_tensor = self.tensorFromSMILES(
            self.output_lang, self.pairs[idx][1])
        condition_tensor = self.conditions[idx]
        return input_tensor, target_tensor, condition_tensor

    @staticmethod
    def tensorFromSMILES(lang, smiles):
        """
        Convert a SMILES string to a tensor of token indices.

        Args:
            lang (Lang): Language object containing character-to-index mapping
            smiles (str): SMILES string to convert

        Returns:
            torch.Tensor: Tensor of token indices with EOS token appended
        """
        indexes = [lang.char2index[char] for char in smiles]
        indexes.append(EOS_token)  # Append EOS token
        return torch.tensor(indexes, dtype=torch.long)  # Shape: [seq_len]


def collate_fn(batch):
    """
    Custom collate function for batching SMILES data with variable lengths.

    This function handles the batching of variable-length SMILES sequences by:
    1. Adding SOS tokens to input sequences
    2. Padding sequences to the same length within the batch
    3. Converting condition vectors to tensors
    4. Moving all tensors to the appropriate device

    Args:
        batch (list): List of (input_tensor, target_tensor, condition_tensor) tuples

    Returns:
        tuple: (input_tensors, input_lengths, target_tensors, target_lengths, condition_tensors)
            - input_tensors: Padded input sequences with SOS tokens
            - input_lengths: Original sequence lengths before padding
            - target_tensors: Padded target sequences
            - target_lengths: Original target sequence lengths
            - condition_tensors: Batch of condition vectors
    """
    input_tensors, target_tensors, condition_tensors = zip(*batch)

    # Calculate lengths before padding
    input_lengths = [len(tensor) for tensor in input_tensors]
    target_lengths = [len(tensor) for tensor in target_tensors]

    # Move tensors to the correct device and add SOS_token for input sequences
    input_tensors = [torch.cat([torch.tensor([SOS_token], dtype=torch.long, device=device),
                                tensor[:-1].to(device)], dim=0) for tensor in input_tensors]

    # Pad sequences to the same length within the batch
    input_tensors = torch.nn.utils.rnn.pad_sequence(
        input_tensors, padding_value=PAD_token, batch_first=True)
    target_tensors = torch.nn.utils.rnn.pad_sequence(
        target_tensors, padding_value=PAD_token, batch_first=True)

    # Convert lengths and conditions to tensors
    input_lengths = torch.tensor(
        input_lengths, dtype=torch.long, device=device)
    target_lengths = torch.tensor(
        target_lengths, dtype=torch.long, device=device)
    condition_tensors = torch.tensor(
        np.array(condition_tensors), dtype=torch.float, device=device)

    return input_tensors.to(device), input_lengths.to(device), target_tensors.to(device), target_lengths.to(
        device), condition_tensors.to(device)
