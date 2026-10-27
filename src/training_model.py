import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from configs import (MODEL_NAME,NUM_TRAIN_EPOCHS,PER_DEVICE_TRAIN_BATCH_SIZE,GRADIENT_ACCUMULATION_STEPS,LEARNING_RATE,MAX_LENGTH,OUTPUT_DIR)
from lora_config import lora_config
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "lei_n_15_022_2024.json"

with open(DATASET_PATH, "r", encoding="utf-8") as arquivo:
    dados = json.load(arquivo)

dados_formatados = [
    {
        "messages": [
            {"role": "user", "content": exemplo["prompt"]},
            {"role": "assistant", "content": exemplo["response"]},
        ]
    }
    for exemplo in dados
]

dataset = Dataset.from_list(dados_formatados)
print(f"Exemplos carregados: {len(dataset)}")

# Carregando o tokenizer e o modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Configura o treinamento
training_args = SFTConfig(

    output_dir = str(OUTPUT_DIR),
    num_train_epochs = NUM_TRAIN_EPOCHS,
    per_device_train_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps = GRADIENT_ACCUMULATION_STEPS,
    learning_rate = LEARNING_RATE,
    max_length = MAX_LENGTH,
    save_strategy = "epoch",
    logging_steps = 10,
    report_to = "none",
    overwrite_output_dir = True,

)

# Criando o treinador
trainer = SFTTrainer(model = model, args = training_args, train_dataset = dataset, processing_class = tokenizer, peft_config = lora_config)
print("\nVerificação dos parâmetros treináveis:")
trainer.model.print_trainable_parameters()

if "--check-only" in sys.argv:
    print("\nVerificação concluída. O treinamento não será iniciado.")
    sys.exit(0)

print("Treinador configurado com sucesso.")
print(f"Dispositivo MPS disponível: {torch.backends.mps.is_available()}")
print("Pronto para iniciar o fine-tuning.")

# iniciando o processo de fine-tuning
print("\nIniciando o fine-tuning...")
trainer.train()
trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print("\nFine-tuning concluído!")
print(f"Modelo salvo em: {OUTPUT_DIR}")
