import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

def main():

    print(f"Carregando o modelo: {MODEL_NAME}")
    print(f"Dispositivo MPS disponível: {torch.backends.mps.is_available()}")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Dispositivo selecionado: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype = torch.float16 if device == "mps" else torch.float32)
    model.to(device)
    model.eval()

    messages = [
        {
            "role": "system",
            "content": "Você é um assistente útil que responde em português.",
        },
        {
            "role": "user",
            "content": "Me resuma a história do Fluminense.",
        },
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize = False,add_generation_prompt = True)
    inputs = tokenizer(prompt,return_tensors="pt").to(device)
    print("\nGerando resposta...\n")

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens = 256, do_sample = False,pad_token_id = tokenizer.eos_token_id)
    generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

    response = tokenizer.decode(generated_tokens,skip_special_tokens = True)
    print("Resposta do modelo:")
    print(response)

if __name__ == "__main__":
    main()

