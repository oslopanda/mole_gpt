import torch.nn.functional as F
import torch
from model import ConditionalTransformerDecoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PAD_token = 0
SOS_token = 1
EOS_token = 2

output_size = 66  # Vocabulary size
hidden_size = 512
condition_vector_size = 6
num_heads = 16
num_layers = 24
max_seq_length = 220
dropout = 0.1

index2char = {0: 'PAD', 1: 'SOS', 2: 'EOS', 3: 'C', 4: '=', 5: '(', 6: 'O', 7: ')', 8: 'N', 9: 'P', 10: '[', 11: '-', 12: ']', 13: 'n', 14: '1', 15: 'c', 16: 'F', 17: 'S', 18: 'H', 19: '+', 20: '2', 21: '3', 22: 'o', 23: '4', 24: '5', 25: '6', 26: '#', 27: 'l', 28: 's', 29: 'i', 30: 'I', 31: 'B', 32: 'r', 33: '7', 34: '8', 35: 'e', 36: 'A', 37: 'G', 38: '9', 39: '%', 40: '0', 41: 'p', 42: 'T', 43: 'b', 44: 't', 45: 'W', 46: 'Z', 47: 'g', 48: 'a', 49: 'd', 50: 'R', 51: 'u', 52: 'E', 53: 'M', 54: 'L', 55: 'X', 56: 'm', 57: 'V', 58: 'f', 59: 'U', 60: 'h', 61: 'Y', 62: 'K', 63: 'k', 64: 'D', 65: 'y'}
decoder_model=ConditionalTransformerDecoder(output_size, hidden_size, condition_vector_size, num_heads, num_layers,
                                              max_seq_length, dropout).to(device)
decoder_model.load_state_dict(torch.load('condition_GPT_77M_6C.pth'))
decoder_model.eval()
def generate_sequence(decoder_model, start_token, max_length, condition_vector, device, temperature=0.5):
    decoder_model.eval()  # Set the model to evaluation mode
    input_seq = torch.tensor([[start_token]], dtype=torch.long, device=device)
    generated_sequence = []

    # Ensure the condition vector is on the same device as the model and is a float tensor
    condition_vector = condition_vector.to(device).float()

    for _ in range(max_length):
        with torch.no_grad():
            # Pass both the input sequence and the condition vector to the decoder model
            output = decoder_model(input_seq, torch.tensor([input_seq.size(0)], device=device), condition_vector)
            output = output / temperature  # Adjust softmax temperature
            probabilities = F.softmax(output[-1, :], dim=-1)

            # Sample from the probability distribution
            next_token = torch.multinomial(probabilities, 1).item()

        generated_sequence.append(next_token)

        if next_token == EOS_token:
            break

        next_token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
        input_seq = torch.cat([input_seq, next_token_tensor], dim=0)

    return generated_sequence


def sequence_to_string(sequence, index2char):
    """
    Convert a sequence of token IDs back to a string using the index2char mapping
    Args:
        sequence: A list of token IDs representing the generated sequence.
        index2char: A dictionary mapping from token IDs to characters.
    Returns:
        A string representing the decoded sequence.
    """
    # Map each token ID to its corresponding character
    decoded_chars = [index2char[token_id] for token_id in sequence if token_id in index2char]

    # Join the characters into a single string
    decoded_string = ''.join(decoded_chars)

    return decoded_string

condition = torch.tensor([365,25,3,3,0.8,3]).to(device)

gen = generate_sequence(decoder_model, start_token=SOS_token, max_length=100, condition_vector=condition, device=device, temperature=0.5)[:-1]
decoded_string = sequence_to_string(gen, index2char)
print("Decoded sequence:", decoded_string)