"""
GPT4mole: Training Script for Conditional Molecular Generation

This script trains a conditional transformer model on molecular SMILES data
with associated molecular properties. The model learns to generate molecules
with desired characteristics.

Author: Your Name
License: MIT
"""

from model import ConditionalTransformerDecoder, masked_cross_entropy
from dataset_making import SMILESDataset, collate_fn
from helping_functions import make_pairs, clean_data, prepareData
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from tqdm.auto import tqdm
import pandas as pd
import torch
import random
random.seed(41)  # Set random seed for reproducibility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data filtering parameters
MIN_LENGTH = 3    # Minimum SMILES string length
MAX_LENGTH = 200  # Maximum SMILES string length (covers >99% of molecules)

# Training configuration
test = False        # Set to True for testing with smaller dataset
saving_pth = False  # Set to True to save the trained model
num_epochs = 2      # Number of training epochs


# Load and prepare data
if test:
    print('Please wait for data to be loaded... (may take a while)')
    all_df = pd.read_parquet('data/smiles_77M_6_features.parquet')
    print('Parquet file loaded')
    all_df = all_df[:2000000]  # Use subset for testing
    print('Test data selected (2M samples)')
else:
    print('Please wait for data to be loaded... (may take a while)')
    all_df = pd.read_parquet('data/smiles_77M_6_features.parquet')
    print('Full dataset loaded')

# Clean and prepare data
df_cleaned = clean_data(all_df)
print('Data cleaning completed')
print('Preparing data pairs...')
data_pairs = make_pairs(df_cleaned)
print('Data pairs created')

# Prepare training data with vocabulary building
print('Building vocabularies and filtering sequences...')
input_lang, output_lang, pairs, conditions = prepareData(
    "smiles", "smiles", data_pairs, MAX_LENGTH=MAX_LENGTH, MIN_LENGTH=MIN_LENGTH)

# Display sample data for verification
print("Sample training pair:", random.choice(pairs))
print("Input vocabulary size:", input_lang.n_chars)
print("Sample condition vector:", random.choice(conditions))

# Model hyperparameters (must match generation script)
output_size = output_lang.n_chars  # Vocabulary size from data
hidden_size = 512                  # Hidden dimension
condition_vector_size = 6          # Number of molecular properties
num_heads = 16                     # Number of attention heads
num_layers = 24                    # Number of transformer layers
max_seq_length = 220               # Maximum sequence length
dropout = 0.1                      # Dropout rate for regularization

# Create dataset and data loader
dataset = SMILESDataset(pairs, conditions, input_lang, output_lang)
data_loader = DataLoader(dataset, batch_size=40,
                         shuffle=True, collate_fn=collate_fn)

# Initialize the conditional transformer model
decoder_model = ConditionalTransformerDecoder(output_size, hidden_size, condition_vector_size, num_heads, num_layers,
                                              max_seq_length, dropout)
decoder_model = decoder_model.to(device)


def count_parameters(model):
    """
    Count the number of trainable parameters in the given model.

    Args:
        model (torch.nn.Module): The model to count parameters for

    Returns:
        int: The total number of trainable parameters in the model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Display model information
total_params = count_parameters(decoder_model)
print(f"Total trainable parameters: {total_params:,}")

# Initialize training components
scaler = GradScaler()  # For mixed precision training
optimizer = optim.Adam(decoder_model.parameters(),
                       lr=0.00001)  # Very low learning rate
# Aggressive learning rate decay
scheduler = StepLR(optimizer, step_size=1, gamma=0.2)

# Set logging frequency based on dataset size
if test:
    print_interval = 1000   # More frequent logging for testing
else:
    print_interval = 10000  # Less frequent logging for full training

# Training loop
print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    total_loss = 0
    data_loader_tqdm = tqdm(enumerate(data_loader),
                            total=len(data_loader), leave=False)

    for batch_idx, (input_batch, input_lengths, target_batch, target_lengths, condition_batch) in data_loader_tqdm:
        # Transpose to (seq_length, batch_size) format expected by model
        input_batch = input_batch.transpose(0, 1)
        target_batch = target_batch.transpose(0, 1)

        optimizer.zero_grad()

        # Forward pass with mixed precision
        with autocast():
            output = decoder_model(input_batch, input_lengths, condition_batch)
            loss = masked_cross_entropy(output, target_batch, target_lengths)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Print intermediate training metrics
        if (batch_idx + 1) % print_interval == 0:
            avg_loss = total_loss / print_interval
            tqdm.write(
                f'Epoch {epoch}, Batch {batch_idx + 1}/{len(data_loader)}, Loss: {avg_loss:.4f}')
            total_loss = 0  # Reset total loss for the next set of batches

    # Calculate and display epoch statistics
    avg_epoch_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch} completed, Avg Loss: {avg_epoch_loss:.4f}')

    # Update learning rate
    scheduler.step()

# Save the trained model if requested
if saving_pth:
    torch.save(decoder_model.state_dict(), 'condition_GPT_77M_6C.pth')
    print("Model saved successfully!")
