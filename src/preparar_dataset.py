import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "lei_n_15_022_2024.json"

with open(DATASET_PATH, "r", encoding="utf-8") as arquivo:
    dataset = json.load(arquivo)

def formatar_conversa(exemplo):
    return {
        "messages": [
            {"role": "user", "content": exemplo["prompt"]},
            {"role": "assistant", "content": exemplo["response"]}
        ]
    }

dataset_formatado = [formatar_conversa(exemplo) for exemplo in dataset]
print(f"Quantidade de exemplos: {len(dataset_formatado)}")
print("\nPrimeira conversa:")
print(json.dumps(dataset_formatado[0], ensure_ascii = False, indent = 2))
