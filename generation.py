"""
GPT4mole: Molecular Generation Script

This script demonstrates how to use the trained conditional transformer model
to generate new SMILES strings based on specified molecular properties.

Author: Wen Xing
License: MIT
"""

import torch.nn.functional as F
import torch
from model import ConditionalTransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Special tokens for sequence processing
PAD_token = 0  # Padding token
SOS_token = 1  # Start-of-sequence token
EOS_token = 2  # End-of-sequence token

# Model hyperparameters (must match training configuration)
output_size = 66  # Vocabulary size
hidden_size = 512
condition_vector_size = 6  # Number of molecular properties
num_heads = 16
num_layers = 24
max_seq_length = 220
dropout = 0.1

# Vocabulary mapping from token indices to characters
# This must match the vocabulary used during training
index2char = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'C', 4: '=', 5: '(', 6: 'O', 7: ')', 8: 'N', 9: 'P', 10: '[', 11: '-', 12: ']', 13: 'n', 14: '1', 15: 'c', 16: 'F', 17: 'S', 18: 'H', 19: '+', 20: '2', 21: '3', 22: 'o', 23: '4', 24: '5', 25: '6', 26: '#', 27: 'l', 28: 's', 29: 'i', 30: 'I', 31: 'B', 32: 'r',
              33: '7', 34: '8', 35: 'e', 36: 'A', 37: 'G', 38: '9', 39: '%', 40: '0', 41: 'p', 42: 'T', 43: 'b', 44: 't', 45: 'W', 46: 'Z', 47: 'g', 48: 'a', 49: 'd', 50: 'R', 51: 'u', 52: 'E', 53: 'M', 54: 'L', 55: 'X', 56: 'm', 57: 'V', 58: 'f', 59: 'U', 60: 'h', 61: 'Y', 62: 'K', 63: 'k', 64: 'D', 65: 'y'}

# Initialize and load the pre-trained model
decoder_model = ConditionalTransformerDecoder(output_size, hidden_size, condition_vector_size, num_heads, num_layers,
                                              max_seq_length, dropout).to(device)
decoder_model.load_state_dict(torch.load('condition_GPT_77M_6C.pth'))
decoder_model.eval()  # Set to evaluation mode


def generate_sequence(decoder_model, start_token, max_length, condition_vector, device, temperature=0.5):
    """
    Generate a SMILES sequence using the conditional transformer model.

    This function performs autoregressive generation, producing one token at a time
    based on the previous tokens and the specified molecular properties.

    Args:
        decoder_model (ConditionalTransformerDecoder): The trained model
        start_token (int): Token to start generation (typically SOS_token)
        max_length (int): Maximum number of tokens to generate
        condition_vector (torch.Tensor): Molecular properties to condition on
        device (torch.device): Device to run inference on
        temperature (float): Temperature for sampling (lower = more deterministic)

    Returns:
        list: Generated sequence of token indices (excluding EOS token)
    """
    decoder_model.eval()  # Set the model to evaluation mode
    input_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)
    generated_sequence = []

    # Ensure the condition vector is on the same device as the model and is a float tensor
    condition_vector = condition_vector.to(device).float()

    for _ in range(max_length):
        with torch.no_grad():
            # Pass both the input sequence and the condition vector to the decoder model
            output = decoder_model(input_seq, torch.tensor(
                [input_seq.size(0)], device=device), condition_vector)
            output = output / temperature  # Adjust softmax temperature
            probabilities = F.softmax(output[-1, :], dim=-1)

            # Sample from the probability distribution
            next_token = torch.multinomial(probabilities, 1).item()

        generated_sequence.append(next_token)

        if next_token == EOS_token:
            break

        next_token_tensor = torch.tensor(
            [[next_token]], dtype=torch.long, device=device)
        input_seq = torch.cat([input_seq, next_token_tensor], dim=0)

    return generated_sequence


def sequence_to_string(sequence, index2char):
    """
    Convert a sequence of token IDs back to a SMILES string.

    This function decodes the numerical token sequence back into a readable
    SMILES string using the vocabulary mapping.

    Args:
        sequence (list): List of token IDs representing the generated sequence
        index2char (dict): Dictionary mapping from token IDs to characters

    Returns:
        str: Decoded SMILES string
    """
    # Map each token ID to its corresponding character
    decoded_chars = [index2char[token_id]
                     for token_id in sequence if token_id in index2char]

    # Join the characters into a single string
    decoded_string = ''.join(decoded_chars)

    return decoded_string


# Example usage: Generate a molecule with specific properties
# Condition vector contains 6 molecular properties (adjust values as needed)
# The exact meaning of each feature depends on your training data
condition = torch.tensor([365, 25, 3, 3, 0.8, 3]).to(device)

# Generate a new SMILES string
gen = generate_sequence(decoder_model, start_token=SOS_token, max_length=100,
                        condition_vector=condition, device=device, temperature=0.5)[:-1]
decoded_string = sequence_to_string(gen, index2char)
print("Generated SMILES:", decoded_string)
