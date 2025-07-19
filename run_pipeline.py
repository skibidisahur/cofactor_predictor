import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to PDB or FASTA file")
args = parser.parse_args()

input_path = args.input
filename = os.path.basename(input_path)
base, ext = os.path.splitext(filename)

if ext.lower() in [".fa", ".fasta"]:
    print("[🧬] Detected FASTA file, running ESMFold...")
    subprocess.run(["python", "esmfold_predict_structure.py", "--fasta", input_path, "--out", f"{base}.pdb"])
    input_path = f"{base}.pdb"
    base = base

print("[📦] Running fpocket...")
subprocess.run(["bash", "run_fpocket.sh", input_path])

print("[🔬] Extracting features...")
subprocess.run(["python", "extract_features.py", "--pdb_id", base])

print("[🤖] Running prediction...")
subprocess.run(["python", "predict_ligand.py", "--features", f"pockets_{base}.csv"])