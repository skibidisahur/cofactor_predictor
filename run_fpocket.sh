if [ -z "$1" ]; then
  echo "Usage: bash run_fpocket.sh path/to/structure.pdb"
  exit 1
fi

fpocket -f "$1"
echo "[+] fpocket completed for $1"