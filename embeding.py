"""
GPT4mole: Molecular Embedding and Property Calculation

This module provides functionality for extracting molecular embeddings from the trained
conditional transformer model and computing molecular properties.

Author: Wen Xing
License: MIT
"""

import torch
import torch.nn.functional as F
import numpy as np
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit.Contrib.SA_Score import sascorer
from model import ConditionalTransformerDecoder, generate_square_subsequent_mask
from generation import index2char  # Import the vocabulary mapping

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration (must match your trained model)
output_size = 66
hidden_size = 512
condition_vector_size = 6
num_heads = 16
num_layers = 24
max_seq_length = 220
dropout = 0.1


def compute_selected_descriptors(smiles, selected_descriptors=['MolWt', 'HeavyAtomCount', 'RingCount', 'MolLogP', 'qed']):
    """
    Compute selected RDKit molecular descriptors.

    Args:
        smiles (str): SMILES string of the molecule
        selected_descriptors (list): List of descriptor names to compute

    Returns:
        dict: Dictionary of descriptor names and values
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        descriptors = {desc_name: Descriptors.__dict__.get(
            desc_name)(mol) for desc_name in selected_descriptors}
        return descriptors
    else:
        return {desc_name: None for desc_name in selected_descriptors}


def calculate_sascore(smiles):
    """
    Calculate the Synthetic Accessibility Score for a molecule.

    Args:
        smiles (str): SMILES string of the molecule

    Returns:
        float: SA Score or None if calculation fails
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return sascorer.calculateScore(mol)
        else:
            return None
    except Exception as e:
        print(f"Error processing SMILES {smiles}: {str(e)}")
        return None


def compute_6_properties(smiles):
    """
    Compute the 6 molecular properties used as conditions in your model.

    Args:
        smiles (str): SMILES string of the molecule

    Returns:
        list: List of 6 molecular properties [MolWt, HeavyAtomCount, RingCount, MolLogP, qed, sascore]
    """
    # Get the 5 RDKit descriptors
    descriptors = compute_selected_descriptors(smiles)

    # Get SA Score
    sascore = calculate_sascore(smiles)

    # Combine into the 6-property vector
    if all(v is not None for v in descriptors.values()) and sascore is not None:
        properties = [
            descriptors['MolWt'],
            descriptors['HeavyAtomCount'],
            descriptors['RingCount'],
            descriptors['MolLogP'],
            descriptors['qed'],
            sascore
        ]
        return properties
    else:
        return None


def smiles_to_tensor(smiles, char2index):
    """
    Convert SMILES string to tensor of token indices.

    Args:
        smiles (str): SMILES string
        char2index (dict): Character to index mapping

    Returns:
        torch.Tensor: Tensor of token indices
    """
    try:
        indexes = [char2index[char] for char in smiles if char in char2index]
        indexes.append(2)  # EOS token
        return torch.tensor(indexes, dtype=torch.long, device=device)
    except KeyError as e:
        print(f"Unknown character in SMILES: {e}")
        return None


def extract_molecular_embedding(decoder_model, smiles, char2index, use_conditions=False, condition_vector=None, pooling_strategy='mean'):
    """
    Extract molecular embedding from the trained conditional transformer model.

    Args:
        decoder_model: Trained ConditionalTransformerDecoder model
        smiles (str): SMILES string of the molecule
        char2index (dict): Character to index mapping from training
        use_conditions (bool): Whether to use condition vector in embedding
        condition_vector (torch.Tensor, optional): 6-dimensional condition vector
        pooling_strategy (str): Pooling method ('mean', 'max', 'last', 'std', 'concat')

    Returns:
        torch.Tensor: Molecular embedding vector of shape (hidden_size,)
    """
    decoder_model.eval()

    # Convert SMILES to tensor
    input_tensor = smiles_to_tensor(smiles, char2index)
    if input_tensor is None:
        return None

    input_seq = input_tensor.unsqueeze(1)  # Shape: (seq_len, 1)

    with torch.no_grad():
        if use_conditions and condition_vector is not None:
            # Use provided condition vector
            condition_vector = condition_vector.to(
                device).float().unsqueeze(0)  # Shape: (1, 6)
        else:
            # Use zero condition vector (for property prediction tasks)
            condition_vector = torch.zeros(1, 6, device=device)

        # Extract embeddings step by step
        # 1. Token embeddings
        token_embeddings = decoder_model.embedding(input_seq)

        # 2. Condition embeddings
        condition_emb = decoder_model.condition_embedding_net(condition_vector)

        # 3. Combine embeddings
        if use_conditions:
            combined_embeddings = token_embeddings + condition_emb.unsqueeze(0)
        else:
            # For property prediction, don't add condition information
            combined_embeddings = token_embeddings

        # 4. Add positional encoding
        positioned_embeddings = decoder_model.positional_encoding(
            combined_embeddings)

        # 5. Process through transformer layers
        tgt_mask = generate_square_subsequent_mask(
            input_seq.size(0)).to(device)
        output = positioned_embeddings

        for layer in decoder_model.layers:
            output = layer(output, tgt_mask)

        # 6. Global pooling to get molecule-level representation
        if pooling_strategy == 'mean':
            molecular_embedding = output.mean(dim=0).squeeze()
        elif pooling_strategy == 'max':
            molecular_embedding = output.max(dim=0)[0].squeeze()
        elif pooling_strategy == 'last':
            molecular_embedding = output[-1].squeeze()
        elif pooling_strategy == 'std':
            molecular_embedding = output.std(dim=0).squeeze()
        elif pooling_strategy == 'concat':
            # Concatenate mean and max pooling for richer representation
            mean_pool = output.mean(dim=0).squeeze()
            max_pool = output.max(dim=0)[0].squeeze()
            molecular_embedding = torch.cat([mean_pool, max_pool], dim=0)
        elif pooling_strategy == 'std_mean':
            # Concatenate std and mean pooling (best of both worlds)
            mean_pool = output.mean(dim=0).squeeze()
            std_pool = output.std(dim=0).squeeze()
            molecular_embedding = torch.cat([mean_pool, std_pool], dim=0)
        elif pooling_strategy == 'all':
            # Concatenate all pooling strategies
            mean_pool = output.mean(dim=0).squeeze()
            max_pool = output.max(dim=0)[0].squeeze()
            std_pool = output.std(dim=0).squeeze()
            last_pool = output[-1].squeeze()
            molecular_embedding = torch.cat(
                [mean_pool, max_pool, std_pool, last_pool], dim=0)
        else:
            raise ValueError(
                "pooling_strategy must be one of: 'mean', 'max', 'last', 'std', 'concat', 'std_mean', 'all'")

    return molecular_embedding


def load_trained_model(model_path='condition_GPT_77M_6C.pth'):
    """
    Load the trained conditional transformer model.

    Args:
        model_path (str): Path to the trained model file

    Returns:
        ConditionalTransformerDecoder: Loaded model
    """
    model = ConditionalTransformerDecoder(
        output_size, hidden_size, condition_vector_size,
        num_heads, num_layers, max_seq_length, dropout
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def create_char2index_from_generation():
    """
    Create char2index mapping from the index2char in generation.py

    Returns:
        dict: Character to index mapping
    """
    char2index = {char: idx for idx, char in index2char.items()}
    return char2index


class MolecularEmbeddingExtractor:
    """
    Class for extracting molecular embeddings and computing properties.
    """

    def __init__(self, model_path='condition_GPT_77M_6C.pth'):
        """
        Initialize the embedding extractor.

        Args:
            model_path (str): Path to the trained model
        """
        self.model = load_trained_model(model_path)
        self.char2index = create_char2index_from_generation()

    def get_embedding(self, smiles, use_conditions=False, condition_vector=None, pooling_strategy='mean'):
        """
        Get molecular embedding for a SMILES string.

        Args:
            smiles (str): SMILES string
            use_conditions (bool): Whether to include condition information
            condition_vector (torch.Tensor, optional): Condition vector
            pooling_strategy (str): Pooling method ('mean', 'max', 'last', 'std', 'concat')

        Returns:
            torch.Tensor: Molecular embedding
        """
        return extract_molecular_embedding(
            self.model, smiles, self.char2index,
            use_conditions, condition_vector, pooling_strategy
        )

    def get_properties(self, smiles):
        """
        Compute the 6 molecular properties for a SMILES string.

        Args:
            smiles (str): SMILES string

        Returns:
            list: List of 6 properties or None if computation fails
        """
        return compute_6_properties(smiles)

    def get_embedding_with_computed_properties(self, smiles):
        """
        Get molecular embedding using computed properties as conditions.

        Args:
            smiles (str): SMILES string

        Returns:
            tuple: (embedding, properties) or (None, None) if failed
        """
        properties = self.get_properties(smiles)
        if properties is None:
            return None, None

        condition_vector = torch.tensor(properties, dtype=torch.float)
        embedding = self.get_embedding(
            smiles, use_conditions=True, condition_vector=condition_vector)

        return embedding, properties

    def compute_similarity(self, smiles1, smiles2, similarity_metric='euclidean', pooling_strategy='mean'):
        """
        Compute similarity between two molecules based on their embeddings.

        Args:
            smiles1 (str): First SMILES string
            smiles2 (str): Second SMILES string
            similarity_metric (str): 'cosine' or 'euclidean'
            pooling_strategy (str): Pooling method for embeddings

        Returns:
            float: Similarity score
        """
        emb1 = self.get_embedding(
            smiles1, use_conditions=False, pooling_strategy=pooling_strategy)
        emb2 = self.get_embedding(
            smiles2, use_conditions=False, pooling_strategy=pooling_strategy)

        if emb1 is None or emb2 is None:
            return None

        if similarity_metric == 'cosine':
            similarity = F.cosine_similarity(
                emb1.unsqueeze(0), emb2.unsqueeze(0))
            return similarity.item()
        elif similarity_metric == 'euclidean':
            distance = torch.norm(emb1 - emb2)
            # Convert distance to similarity
            return 1.0 / (1.0 + distance.item())
        else:
            raise ValueError(
                "similarity_metric must be 'cosine' or 'euclidean'")


def example_usage():
    """
    Example of how to use the embedding extractor.
    """
    # Initialize the extractor
    extractor = MolecularEmbeddingExtractor()

    # Example SMILES
    smiles = "CCO"  # Ethanol

    # Get embedding without conditions (for property prediction)
    embedding_no_cond = extractor.get_embedding(smiles, use_conditions=False)
    print(f"Embedding shape (no conditions): {embedding_no_cond.shape}")

    # Get properties
    properties = extractor.get_properties(smiles)
    print(f"Computed properties: {properties}")

    # Get embedding with computed properties
    embedding_with_cond, _ = extractor.get_embedding_with_computed_properties(
        smiles)
    print(f"Embedding shape (with conditions): {embedding_with_cond.shape}")

    # Compare two molecules with different pooling strategies
    smiles2 = "CCO"  # Same molecule
    smiles3 = "CCCCCCCF"  # Different molecule

    print("\n=== Testing Different Pooling Strategies ===")
    pooling_strategies = ['mean', 'max', 'last', 'std', 'concat', 'std_mean']

    for strategy in pooling_strategies:
        similarity_same = extractor.compute_similarity(
            smiles, smiles2, pooling_strategy=strategy)
        similarity_different = extractor.compute_similarity(
            smiles, smiles3, pooling_strategy=strategy)

        print(f"{strategy.upper()} pooling:")
        print(f"  Same molecule: {similarity_same:.6f}")
        print(f"  Different molecule: {similarity_different:.6f}")
        print(
            f"  Difference: {abs(similarity_same - similarity_different):.6f}")
        print()

    # Test with very different molecules
    print("=== Testing with Very Different Molecules ===")
    test_molecules = [
        ("CCO", "Ethanol"),
        ("c1ccccc1", "Benzene"),
        ("CC(=O)O", "Acetic acid"),
        ("CCCCCCCCCCCCCCCCC(=O)O", "Stearic acid"),
        ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen")
    ]

    base_smiles = "CCO"
    for smiles, name in test_molecules:
        sim_mean = extractor.compute_similarity(
            base_smiles, smiles, pooling_strategy='mean')
        sim_concat = extractor.compute_similarity(
            base_smiles, smiles, pooling_strategy='concat')
        print(
            f"Similarity to {name}: mean={sim_mean:.6f}, concat={sim_concat:.6f}")


if __name__ == "__main__":
    example_usage()
