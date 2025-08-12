"""
GPT4mole: Helper Functions for Molecular Data Processing

This module contains utility functions and classes for processing molecular data,
including SMILES string handling, data cleaning, and language modeling utilities.

Author: Wen Xing
License: MIT
"""

from rdkit import Chem
from tqdm.auto import tqdm


class Lang:
    """
    Language class for converting between characters and indices in SMILES strings.

    This class maintains vocabularies for character-level tokenization of SMILES strings,
    providing bidirectional mapping between characters and numerical indices.
    Essential for neural network processing of molecular representations.

    Attributes:
        name (str): Name identifier for this language
        char2index (dict): Mapping from characters to indices
        char2count (dict): Character frequency counts
        index2char (dict): Mapping from indices to characters
        n_chars (int): Total number of unique characters in vocabulary
    """

    def __init__(self, name):
        """
        Initialize the language object with special tokens.

        Args:
            name (str): Name identifier for this language (e.g., "smiles")
        """
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}  # Special tokens
        self.n_chars = 3  # Count SOS, EOS, and PAD tokens

    def addSMILES(self, smiles):
        """
        Add all characters from a SMILES string to the vocabulary.

        Args:
            smiles (str): SMILES string representation of a molecule
        """
        for char in smiles:
            self.addChar(char)

    def addChar(self, char):
        """
        Add a single character to the vocabulary or increment its count.

        Args:
            char (str): Character to add to vocabulary
        """
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1


def canonicalize(smiles):
    """
    Convert a SMILES string to its canonical form.

    Takes a SMILES representation of a chemical and returns a standardized
    canonical SMILES representation using RDKit. Use this if your dataset
    is not canonicalized to ensure consistent molecular representations.

    Args:
        smiles (str): SMILES string representation of a molecule

    Returns:
        str: Canonical SMILES string
    """
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def validSMILES(smiles):
    """
    Check if a given SMILES string represents a valid molecule.

    Validates a SMILES representation using RDKit's molecular parser.
    This function is useful for validating generated molecules but is
    not used during training.

    Args:
        smiles (str): SMILES string to validate

    Returns:
        bool: True if valid SMILES, False otherwise
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True


def make_pairs(df_cleaned):
    """
    Create training pairs from cleaned molecular data.

    Converts a DataFrame containing SMILES and molecular properties into
    a list of (SMILES, conditions) tuples for training the conditional model.

    Args:
        df_cleaned (pd.DataFrame): Cleaned DataFrame with 'smiles' column and property columns

    Returns:
        list: List of (smiles_string, condition_array) tuples
    """
    smiles_strings = df_cleaned['smiles'].values
    conditions = df_cleaned.drop('smiles', axis=1)
    data_pairs = [(smiles_strings[i], conditions.iloc[i].values)
                  for i in tqdm(range(len(smiles_strings)))]
    return data_pairs


def clean_data(all_df):
    """
    Clean molecular dataset by removing entries with missing values.

    Args:
        all_df (pd.DataFrame): Raw DataFrame containing molecular data

    Returns:
        pd.DataFrame: Cleaned DataFrame with NaN values removed
    """
    df_cleaned = all_df.dropna()
    print(
        f'number of molecules will be used for further processing: {len(df_cleaned)}')
    return df_cleaned


def prepareData(lang1, lang2, data_pairs, MAX_LENGTH, MIN_LENGTH):
    """
    Prepare molecular data for training by creating vocabularies and filtering sequences.

    This function processes raw molecular data pairs to create language objects for
    character-level tokenization and filters sequences by length constraints.

    Args:
        lang1 (str): Name for input language (typically "smiles")
        lang2 (str): Name for output language (typically "smiles")
        data_pairs (list): List of (smiles_string, condition_array) tuples
        MAX_LENGTH (int): Maximum allowed SMILES string length
        MIN_LENGTH (int): Minimum allowed SMILES string length

    Returns:
        tuple: (input_lang, output_lang, filtered_pairs, filtered_conditions)
            - input_lang (Lang): Language object for input sequences
            - output_lang (Lang): Language object for output sequences
            - filtered_pairs (list): Filtered (smiles, smiles) pairs
            - filtered_conditions (list): Corresponding condition vectors
    """
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    print("Read %s smile-condition pairs" % len(data_pairs))

    # Filter pairs and conditions together based on sequence length
    filtered_pairs = []
    filtered_conditions = []
    for smiles, conditions in tqdm(data_pairs):
        if len(smiles) >= MIN_LENGTH and len(smiles) <= MAX_LENGTH:
            # Pairing SMILES with itself for autoregressive training
            filtered_pairs.append((smiles, smiles))
            filtered_conditions.append(conditions)
    print("Filtered to %d pairs" % len(filtered_pairs))

    # Build vocabularies from filtered data
    print("Counting chars...")
    for smiles, _ in tqdm(filtered_pairs):
        input_lang.addSMILES(smiles)
        output_lang.addSMILES(smiles)
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, filtered_pairs, filtered_conditions
