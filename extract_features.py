import os
import argparse
import numpy as np
import pandas as pd
import re
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
import esm
import torch

# Load ESM model
model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
batch_converter = alphabet.get_batch_converter()
model.eval()

three_to_one = IUPACData.protein_letters_3to1
pdb_parser = PDBParser(QUIET=True)

def get_pocket_residues(pocket_path):
    structure = pdb_parser.get_structure("pocket", pocket_path)
    residues = []
    for model in structure:
        for chain in model:
            for residue in chain:
                residues.append(residue)
    return residues

def pocket_sequence(residues):
    seq = ""
    for r in residues:
        aa = three_to_one.get(r.get_resname().capitalize(), "X")
        seq += aa
    return seq

def get_esm_embedding(seq):
    if len(seq) == 0:
        return np.zeros(1280)
    data = [("pocket", seq)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=False)
    rep = results["representations"][33]
    return rep[0, 1:-1].mean(0).numpy()

def parse_fpocket_info(info_path):
    with open(info_path, 'r') as f:
        lines = f.readlines()

    pocket_data = {}
    pocket_id = None

    for line in lines:
        line = line.strip()
        if line.startswith("Pocket"):
            match = re.match(r"Pocket (\d+)", line)
            if match:
                pocket_id = int(match.group(1))
        elif ":" in line:
            key, value = [x.strip() for x in line.split(":", 1)]
            key = key.lower().replace(" ", "_").replace(".", "").replace("-", "_")
            try:
                value = float(value)
            except ValueError:
                continue
            pocket_data[(pocket_id, key)] = value

    return pocket_data

def main(pdb_id):
    pocket_dir = f"{pdb_id}_out/pockets"
    info_path = f"{pdb_id}_out/{pdb_id}_info.txt"
    if not os.path.exists(pocket_dir):
        print(f"[!] No pockets found in {pocket_dir}")
        return

    fpocket_data = parse_fpocket_info(info_path)

    rows = []
    for fname in os.listdir(pocket_dir):
        if not fname.endswith("_atm.pdb"):
            continue
        pocket_path = os.path.join(pocket_dir, fname)
        pocket_id = int(fname.split("pocket")[1].split("_")[0])
        residues = get_pocket_residues(pocket_path)
        seq = pocket_sequence(residues)
        emb = get_esm_embedding(seq)

        row = {
            "pdb_id": pdb_id,
            "pocket_id": pocket_id,
            "pocket_sequence": seq,
        }
        for i, val in enumerate(emb):
            row[f"embedding_{i}"] = val

        # Add fpocket structural data
        for k, v in fpocket_data.items():
            pid, feature = k
            if pid == pocket_id:
                row[feature] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"features_{pdb_id}.csv", index=False)
    print(f"[+] Saved combined ESM + structural features to features_{pdb_id}.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_id", help="PDB ID used for folder names, e.g., 1A8P")
    args = parser.parse_args()
    main(args.pdb_id)