This tool predicts ligand presence and identity for protein pockets using `fpocket`, ESM embeddings, and trained XGBoost models.

## Usage
1. Run esmfold:
```bash
   esmfold_from_sequence.py --sequence path/to/sequence --output pdb_id.pdb
2. Run fpocket:
```bash
bash run_fpocket.sh path/to/your_structure.pdb
```

3. Extract features:
```bash
python extract_features.py --pdb_id your_structure
```

4. Predict ligand binding:
```bash
python predict_ligand.py --features pockets_your_structure.csv
```

## Outputs
- `ligand_predictions.csv`: Contains predicted ligand presence and ligand type per pocket.

---