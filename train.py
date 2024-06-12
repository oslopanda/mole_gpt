import pandas as pd
import torch
import random
random.seed(41)
from tqdm.auto import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from helping_functions import make_pairs, clean_data, prepareData
from dataset_making import SMILESDataset, collate_fn
from model import ConditionalTransformerDecoder, masked_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MIN_LENGTH = 3
# more than 99% molecules are less than 200
MAX_LENGTH = 200

test = False
saving_pth = False
num_epochs = 2


if test:
    print ('Plese wait data to be loaded....(may take a while)')
    all_df = pd.read_parquet('data/smiles_77M_6_features.parquet')
    print('csv file loaded')
    all_df = all_df[:2000000]
    print ('test data selected')
else:
    print ('Plese wait data to be loaded....(may take a while)')
    all_df = pd.read_parquet('data/smiles_77M_6_features.parquet')
    print('csv file loaded')

df_cleaned = clean_data(all_df)
print ('clean data loaded')
print ('preparing data pairs.....')
data_pairs = make_pairs(df_cleaned)
print ('data pairs loaded')

print('Start to prepare training data.....')
input_lang, output_lang, pairs, conditions = prepareData("smiles", "smiles", data_pairs, MAX_LENGTH=MAX_LENGTH, MIN_LENGTH=MIN_LENGTH)
print(random.choice(pairs))
print(input_lang.index2char)
print(input_lang.char2index)

print(output_lang.char2index)
print(random.choice(conditions))

# Example hyperparameters
output_size = output_lang.n_chars  # Vocabulary size
hidden_size = 512
condition_vector_size = 6
num_heads = 16
num_layers = 24
max_seq_length = 220
dropout = 0.1

dataset = SMILESDataset(pairs, conditions, input_lang, output_lang)
data_loader = DataLoader(dataset, batch_size=40, shuffle=True, collate_fn=collate_fn)

decoder_model = ConditionalTransformerDecoder(output_size, hidden_size, condition_vector_size, num_heads, num_layers,
                                              max_seq_length, dropout)
decoder_model = decoder_model.to(device)


def count_parameters(model):
    """
    Count the number of trainable parameters in the given model.
    :param model: The model to count parameters for.
    :type model: torch.nn.Module
    :return: The total number of trainable parameters in the model.
    :rtype: int
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


total_params = count_parameters(decoder_model)
print(f"Total trainable parameters: {total_params}")

scaler = GradScaler()
optimizer = optim.Adam(decoder_model.parameters(), lr=0.00001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.2)

if test:
    print_interval = 1000
else:
    print_interval = 10000

for epoch in range(num_epochs):
    total_loss = 0
    data_loader_tqdm = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    for batch_idx, (input_batch, input_lengths, target_batch, target_lengths, condition_batch) in data_loader_tqdm:
        input_batch = input_batch.transpose(0, 1)  # Change to (seq_length, batch_size)
        target_batch = target_batch.transpose(0, 1)  # Change to (seq_length, batch_size)

        optimizer.zero_grad()

        # Using autocast for automatic mixed precision
        with autocast():
            output = decoder_model(input_batch, input_lengths, condition_batch)
            loss = masked_cross_entropy(output, target_batch, target_lengths)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        total_loss += loss.item()

        # Print intermediate training metrics
        if (batch_idx + 1) % print_interval == 0:
            avg_loss = total_loss / print_interval
            tqdm.write(f'Epoch {epoch}, Batch {batch_idx + 1}/{len(data_loader)}, Loss: {avg_loss:.4f}')
            total_loss = 0  # Reset total loss for the next set of batches

    avg_epoch_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch} completed, Avg Loss: {avg_epoch_loss:.4f}')

    # Step the scheduler to update the learning rate
    scheduler.step()

if saving_pth:
    torch.save(decoder_model.state_dict(), 'condition_GPT_77M_6C.pth')


