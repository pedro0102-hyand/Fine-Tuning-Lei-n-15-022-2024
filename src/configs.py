from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

# Parâmetros do fine-tuning
NUM_TRAIN_EPOCHS = 3 # numero de vezes que passamos pelos dados
PER_DEVICE_TRAIN_BATCH_SIZE = 1 # quantidade de exemplos por lote
GRADIENT_ACCUMULATION_STEPS = 8 # quantidade de passos de gradiente antes de atualizar os pesos
LEARNING_RATE = 2e-5
MAX_LENGTH = 512

# Salvamento do modelo treinado
OUTPUT_DIR = BASE_DIR / "output" / "qwen-lei-15022-2024"
print("Configuração do fine-tuning")
print(f"Modelo: {MODEL_NAME}")
print(f"Épocas: {NUM_TRAIN_EPOCHS}")
print(f"Batch size: {PER_DEVICE_TRAIN_BATCH_SIZE}")
print(f"Acumulação de gradientes: {GRADIENT_ACCUMULATION_STEPS}")
print(f"Taxa de aprendizado: {LEARNING_RATE}")
print(f"Tamanho máximo da sequência: {MAX_LENGTH}")
print(f"Diretório de saída: {OUTPUT_DIR}")
