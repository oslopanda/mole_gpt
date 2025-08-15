# GPT4mole: Conditional Molecular Generation with Transformers

GPT4mole is a conditional transformer-based molecular generator that creates SMILES strings of chemical molecules based on specified molecular properties. The model uses a GPT-like architecture with conditional inputs to generate molecules with desired characteristics.

## 🧬 Features

- **Conditional Generation**: Generate molecules with specific properties (molecular weight, logP, etc.)
- **Transformer Architecture**: Based on proven GPT-style decoder-only transformers
- **Large-Scale Training**: Trained on 77M molecular structures
- **Flexible Conditioning**: 6-dimensional condition vectors for molecular properties
- **Chemical Validity**: Built-in SMILES validation using RDKit

## 🚀 Quick Start

### Installation

1. Clone this repository:
```bash
git clone https://github.com/oslopanda/mole_gpt
cd mole_gpt
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the training dataset and pre-trained model:
   - Dataset: [smiles_77M_6_features.parquet](https://figshare.com/articles/dataset/smiles_77M_6_features_parquetcondition_GPT_77M_6C/26028922)
   - Pre-trained model: [condition_GPT_77M_6C.pth](https://figshare.com/articles/dataset/smiles_77M_6_features_parquetcondition_GPT_77M_6C/26028922)

   Place both files in the repository root directory.

### Basic Usage

#### Generate Molecules
```python
python generation.py
```

#### Train Your Own Model
```python
python train.py
```

#### Use the Master Script
```python
python master.py
```

## 📊 Model Architecture

The model consists of:
- **Conditional Transformer Decoder**: 24 layers, 16 attention heads, 512 hidden dimensions
- **Condition Embedding Network**: 2-layer MLP for molecular property encoding
- **Positional Encoding**: Sinusoidal position embeddings
- **Vocabulary**: 66 tokens covering SMILES characters and special tokens

### Model Parameters
- **Total Parameters**: ~77M
- **Hidden Size**: 512
- **Attention Heads**: 16
- **Layers**: 24
- **Condition Vector Size**: 6
- **Max Sequence Length**: 220

## 🔬 Condition Vector

The model accepts 6-dimensional condition vectors representing molecular properties:
```python
condition = torch.tensor([365, 25, 3, 3, 0.8, 3])
# Adjust these values based on desired molecular properties
```

*Note: The exact meaning of each dimension depends on your training data features.*

## 📁 Project Structure

```
GPT4mole/
├── model.py                 # Core transformer model implementation
├── train.py                 # Training script
├── generation.py            # Molecule generation script
├── helping_functions.py     # Utility functions for data processing
├── dataset_making.py        # PyTorch dataset classes
├── master.py               # Main execution script
├── requirements.txt        # Python dependencies
├── data/                   # Data directory
│   └── smiles_77M_6_features.parquet
├── figures/                # Example figures
└── README.md              # This file
```

## 🛠️ API Reference

### Core Classes

#### `ConditionalTransformerDecoder`
Main model class for conditional molecular generation.

```python
model = ConditionalTransformerDecoder(
    output_size=66,           # Vocabulary size
    hidden_size=512,          # Hidden dimension
    condition_vector_size=6,  # Number of molecular properties
    num_heads=16,            # Attention heads
    num_layers=24,           # Transformer layers
    max_seq_length=220,      # Maximum sequence length
    dropout=0.1              # Dropout rate
)
```

#### `SMILESDataset`
PyTorch dataset for handling SMILES strings with conditions.

```python
dataset = SMILESDataset(pairs, conditions, input_lang, output_lang)
```

### Key Functions

#### `generate_sequence()`
Generate SMILES sequences with specified conditions.

```python
sequence = generate_sequence(
    decoder_model=model,
    start_token=SOS_token,
    max_length=100,
    condition_vector=condition,
    device=device,
    temperature=0.5
)
```

#### `validSMILES()`
Validate generated SMILES strings.

```python
is_valid = validSMILES("CCO")  # Returns True for valid SMILES
```

## 🎯 Training

To train your own model:

1. Prepare your dataset in the required format (SMILES + molecular properties)
2. Adjust hyperparameters in `train.py`
3. Run training:

```python
python train.py
```

### Training Configuration
- **Batch Size**: 40
- **Learning Rate**: 1e-5
- **Optimizer**: Adam
- **Scheduler**: StepLR (γ=0.2)
- **Mixed Precision**: Enabled
- **Loss Function**: Masked Cross-Entropy

## 📈 Performance

- **Dataset Size**: 77M molecules
- **Training Time**: Varies based on hardware
- **Memory Requirements**: ~>12GB GPU memory recommended
- **Generation Speed**: ~1-2 molecules/second on GPU

## 🔍 Examples

### Generate a Specific Molecule Type
```python
import torch
from generation import generate_sequence, sequence_to_string

# Define condition for desired properties
condition = torch.tensor([300, 20, 2, 1, 0.7, 2])  # Adjust as needed

# Generate molecule
sequence = generate_sequence(
    decoder_model, SOS_token, 100, condition, device, temperature=0.8
)
smiles = sequence_to_string(sequence[:-1], index2char)
print(f"Generated SMILES: {smiles}")
```

### Validate Generated Molecules
```python
from helping_functions import validSMILES

generated_smiles = "CCO"
if validSMILES(generated_smiles):
    print("Valid molecule!")
else:
    print("Invalid SMILES")
```

## 🧬 Advanced Usage: Molecular Embeddings

GPT4mole can also extract molecular embeddings for downstream tasks like property prediction, similarity search, and clustering. The `embeding.py` module provides comprehensive functionality for these applications.

### Extract Molecular Embeddings

```python
from embeding import MolecularEmbeddingExtractor

# Initialize the embedding extractor
extractor = MolecularEmbeddingExtractor()

# Extract embedding without conditions (for similarity search)
embedding = extractor.get_embedding("CCO", use_conditions=False)
print(f"Embedding shape: {embedding.shape}")  # torch.Size([512])

# Extract embedding with computed properties
embedding_with_props, properties = extractor.get_embedding_with_computed_properties("CCO")
print(f"Properties: {properties}")  # [MW, HeavyAtomCount, RingCount, LogP, QED, SAScore]
```

### Molecular Similarity Search

```python
# Compare molecular similarity (structure-based)
similarity = extractor.compute_similarity(
    "CCO",           # Ethanol
    "CCCO",          # Propanol
    similarity_metric='euclidean',  # Recommended over cosine
    pooling_strategy='mean'         # Best for chemical interpretation
)
print(f"Similarity: {similarity:.3f}")

# Test with very different molecules
molecules = [
    ("CCO", "Ethanol"),
    ("c1ccccc1", "Benzene"),
    ("CC(=O)O", "Acetic acid"),
    ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "Ibuprofen")
]

base_molecule = "CCO"
for smiles, name in molecules:
    sim = extractor.compute_similarity(base_molecule, smiles)
    print(f"Similarity to {name}: {sim:.3f}")
```

### Property Prediction from Embeddings

```python
import torch
import torch.nn as nn

# Create a property prediction model using pre-trained embeddings
class PropertyPredictor(nn.Module):
    def __init__(self, embedding_extractor, num_properties=6):
        super().__init__()
        self.extractor = embedding_extractor
        self.predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_properties)
        )

    def forward(self, smiles_list):
        embeddings = []
        for smiles in smiles_list:
            emb = self.extractor.get_embedding(smiles, use_conditions=False)
            embeddings.append(emb)

        batch_embeddings = torch.stack(embeddings)
        return self.predictor(batch_embeddings)

# Usage
predictor = PropertyPredictor(extractor)
predicted_props = predictor(["CCO", "CCCO", "c1ccccc1"])
```

### Pooling Strategies for Different Tasks

The embedding extraction supports multiple pooling strategies optimized for different applications:

```python
# Different pooling strategies
pooling_options = {
    'mean': 'Best for similarity search and general use',
    'max': 'Captures key structural features',
    'last': 'Uses final token representation',
    'std': 'Captures sequence variability',
    'concat': 'Combines mean and max pooling',
    'std_mean': 'Combines standard deviation and mean'
}

# Test different strategies
for strategy, description in pooling_options.items():
    embedding = extractor.get_embedding("CCO", pooling_strategy=strategy)
    print(f"{strategy}: {embedding.shape} - {description}")
```

### Similarity Thresholds for Chemical Applications

Based on extensive testing with Euclidean distance and mean pooling:

```python
def interpret_similarity(similarity_score):
    """Interpret molecular similarity scores"""
    if similarity_score > 0.8:
        return "Very similar (same scaffold)"
    elif similarity_score > 0.4:
        return "Moderately similar (related structures)"
    elif similarity_score > 0.2:
        return "Different but chemically related"
    else:
        return "Structurally very different"

# Example usage
sim_score = extractor.compute_similarity("CCO", "c1ccccc1")  # ~0.104
print(f"Ethanol vs Benzene: {interpret_similarity(sim_score)}")
```

### Building a Molecular Database

```python
# Create a searchable molecular database
class MolecularDatabase:
    def __init__(self, extractor):
        self.extractor = extractor
        self.molecules = []
        self.embeddings = []

    def add_molecule(self, smiles, metadata=None):
        embedding = self.extractor.get_embedding(smiles, use_conditions=False)
        if embedding is not None:
            self.molecules.append({'smiles': smiles, 'metadata': metadata})
            self.embeddings.append(embedding)

    def search_similar(self, query_smiles, top_k=5, threshold=0.3):
        query_emb = self.extractor.get_embedding(query_smiles, use_conditions=False)
        similarities = []

        for i, emb in enumerate(self.embeddings):
            sim = 1.0 / (1.0 + torch.norm(query_emb - emb).item())
            if sim >= threshold:
                similarities.append((sim, self.molecules[i]))

        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[0], reverse=True)
        return similarities[:top_k]

# Usage
db = MolecularDatabase(extractor)
db.add_molecule("CCO", {"name": "Ethanol", "use": "Solvent"})
db.add_molecule("CCCO", {"name": "Propanol", "use": "Solvent"})

# Search for similar molecules
results = db.search_similar("CCO", top_k=3)
for similarity, mol_data in results:
    print(f"Similarity: {similarity:.3f}, SMILES: {mol_data['smiles']}")
```

### Key Insights from Embedding Analysis

1. **Euclidean distance** works better than cosine similarity for molecular embeddings
2. **Mean pooling** provides the best chemical interpretation and similarity ranges
3. **Structure-only embeddings** (without conditions) are ideal for similarity search
4. **Conditional embeddings** are better for property prediction tasks
5. The model learned chemically meaningful representations that correlate with structural similarity

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Add docstrings to all functions and classes
- Follow PEP 8 style guidelines
- Include tests for new functionality
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@misc{xing2025gpt,
  title = {GPT-like transformer based conditional molecule generator and a high drug-likeness (QED) dataset generation},
  author = {Wen Xing and Juan Yang},
  year = {2025},
  howpublished = {\url{https://chemrxiv.org/engage/chemrxiv/article-details/677e792bfa469535b9306ea3}},
  note = {Preprint, ChemRxiv, Version 2, posted 8 January 2025, DOI: 10.26434/chemrxiv-2024-tq75v-v2}
}

```

## 🙏 Acknowledgments

- Built with PyTorch and RDKit
- Inspired by the original Transformer architecture
- Dataset processing utilities adapted from molecular ML best practices

> This is part of project **SAM — Systematic Workflows for AI (Artificial Intelligence) in Chemistry and Materials Research**,  
> funded by **SINTEF Industry**.


## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation
- Review example scripts in the repository

---

**Happy molecule generation! 🧪✨**