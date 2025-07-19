# Ligand Binding Pocket Classifier

This tool predicts ligand presence and identity for protein pockets using `fpocket`, ESM embeddings, and trained XGBoost models. It also supports inputting protein **sequences** directly using **ESMFold**.

## Usage

### Option 1: From structure
```bash
python run_pipeline.py --input myprotein.pdb
```

### Option 2: From sequence
```bash
python run_pipeline.py --input myprotein.fasta
```

## Outputs
- `ligand_predictions.csv`: Contains predicted ligand presence and ligand type per pocket.

---
