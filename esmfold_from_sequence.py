# save as esmfold_from_sequence.py
import torch
import esm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--sequence", required=True, help="Protein sequence (as a string or FASTA file)")
parser.add_argument("--out", default="predicted.pdb", help="Output PDB filename")
args = parser.parse_args()

# Load model
model = esm.pretrained.esmfold_v1()
model = model.eval().cuda() if torch.cuda.is_available() else model.eval()

# Handle FASTA input or raw sequence
if args.sequence.endswith(".fasta") or args.sequence.endswith(".fa"):
    with open(args.sequence, "r") as f:
        lines = [l.strip() for l in f.readlines() if not l.startswith(">")]
        sequence = "".join(lines)
else:
    sequence = args.sequence.strip()

# Predict structure
with torch.no_grad():
    output = model.infer_pdb(sequence)

# Save PDB
with open(args.out, "w") as f:
    f.write(output)

print(f"[+] Structure written to {args.out}")
