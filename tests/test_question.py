import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "output/qwen-lei-15022-2024"

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Carregando modelo no dispositivo: {device}")

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model_base, ADAPTER_PATH)
model = model.to(device)
model.eval()

messages = [
    {
        "role": "system",
        "content": (
            "Você é um assistente especializado na Lei nº 15.022/2024. "
            "Responda em português com clareza."
        ),
    },
    {
        "role": "user",
        "content": "Quais são as responsabilidades do fabricante segundo a Lei nº 15.022/2024?",
    },
]

inputs = tokenizer.apply_chat_template(messages, tokenize = True, add_generation_prompt = True, return_tensors = "pt").to(device)

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens = 150, do_sample = False, pad_token_id = tokenizer.eos_token_id)
resposta = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens = True)
print("\nPergunta: Resuma a Lei nº 15.022/2024 em uma frase.")
print(f"\nResposta do modelo:\n{resposta}")
