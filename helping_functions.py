from rdkit import Chem
from tqdm.auto import tqdm

class Lang:
    """
    This is the 'language' class for converting between characters and index
    """
    def __init__(self, name):
        self.name = name
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_chars = 3  # Count SOS and EOS and PAD

    def addSMILES(self, smiles):
        for char in smiles:
            self.addChar(char)

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

def canonicalize(smiles):
    # Takes a SMILES representation of a chemical
    # and returns a standardised canonical SMILES representation.
    # use it if your dataset is not canonicalized
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def validSMILES(smiles):
    # Checks if a given SMILES representation of a chemical is valid.
    # Returns True if it's a valid SMILES string, and False otherwise.
    # This helping function is not used for training but good for validate generated molecules
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return True

def make_pairs(df_cleaned):
    smiles_strings = df_cleaned['smiles'].values
    conditions = df_cleaned.drop('smiles', axis=1)
    data_pairs = [(smiles_strings[i], conditions.iloc[i].values) for i in tqdm(range(len(smiles_strings)))]
    return data_pairs

def clean_data(all_df):
    df_cleaned = all_df.dropna()
    print(f'number of molecules will be used for further processing: {len(df_cleaned)}')
    return df_cleaned

def prepareData(lang1, lang2, data_pairs, MAX_LENGTH, MIN_LENGTH):
    """takes data as list of (smiles, conditions) pairs
    returns the dictionaries for input and output languages, the pairs, and conditions"""
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    print("Read %s smile-condition pairs" % len(data_pairs))

    # Filter pairs and conditions together
    filtered_pairs = []
    filtered_conditions = []
    for smiles, conditions in tqdm(data_pairs):
        if len(smiles) >= MIN_LENGTH and len(smiles) <= MAX_LENGTH:
            filtered_pairs.append((smiles, smiles))  # Pairing SMILES with itself
            filtered_conditions.append(conditions)
    print("Filtered to %d pairs" % len(filtered_pairs))

    print("Counting chars...")
    for smiles, _ in tqdm(filtered_pairs):
        input_lang.addSMILES(smiles)
        output_lang.addSMILES(smiles)
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, filtered_pairs, filtered_conditions
