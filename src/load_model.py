import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Dispositivo selecionado: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = model.to(device)

print("\nModelo e tokenizer carregados com sucesso!")
print(f"Tipo do modelo: {model.__class__.__name__}")
print(f"Tamanho do vocabulário: {len(tokenizer)}")
print(f"Dispositivo do modelo: {model.device}")
