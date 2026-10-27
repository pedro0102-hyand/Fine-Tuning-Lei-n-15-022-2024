from peft import LoraConfig

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ],
    bias="none",
    task_type="CAUSAL_LM",
)

print("Configuração LoRA criada com sucesso!")
print(f"Rank (r): {LORA_R}")
print(f"Alpha: {LORA_ALPHA}")
print(f"Dropout: {LORA_DROPOUT}")
print(f"Camadas-alvo: {lora_config.target_modules}")
