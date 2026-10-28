import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel

MODEL_NAME ="Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "output/qwen-lei-15022-2024"

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model_base = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(model_base, ADAPTER_PATH)
model = model.to(device)
model.eval()

pipe = pipeline("text-generation", model = model, tokenizer = tokenizer, device = device)

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
        "content": "Segundo a Lei nº 15.022/2024 quais sao os limites e regras acerca do teste de produtos em animais ? .",
    },
]

outputs = pipe(messages, max_new_tokens = 512, do_sample = False)
print(outputs[0]["generated_text"][-1]["content"])