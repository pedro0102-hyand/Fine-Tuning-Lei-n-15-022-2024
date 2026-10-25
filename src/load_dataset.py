import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "lei_n_15_022_2024.json"

with open(DATASET_PATH, "r", encoding="utf-8") as arquivo:
    dataset = json.load(arquivo)

print(f"Quantidade de exemplos: {len(dataset)}")
print(f"Campos do primeiro exemplo: {list(dataset[0].keys())}")
print("\nPrimeiro exemplo:")
print(f"Prompt: {dataset[0]['prompt']}")
print(f"Response: {dataset[0]['response']}")
