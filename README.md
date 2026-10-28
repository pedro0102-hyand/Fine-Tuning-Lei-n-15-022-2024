# Fine-Tuning da Lei nº 15.022/2024

Este projeto tem como objetivo adaptar um modelo de linguagem para responder perguntas em português sobre a Lei nº 15.022/2024, com foco em temas como inventário nacional de substâncias químicas, gestão de risco, uso de alternativas ao teste em animais e obrigações de fabricantes e importadores.

A implementação usa a base Qwen/Qwen2.5-0.5B-Instruct, aplica técnica de fine-tuning supervisionado com LoRA (Low-Rank Adaptation) e salva um adapter específico para o domínio legal do projeto.

## Visão geral

O repositório reúne:

- um conjunto de perguntas e respostas em JSON sobre a Lei nº 15.022/2024;
- scripts para carregar, preparar e validar o dataset;
- configuração do modelo base e do fine-tuning;
- treinamento com TRL + PEFT;
- script de inferência para testar o modelo ajustado;
- artefatos treinados salvos em `output/`.

O projeto é uma solução totalmente focada em adaptação de modelos de linguagem para um domínio específico, sem depender de uma interface web complexa. Ele mostra uma pipeline completa de fine-tuning em Python, com o ecossistema Hugging Face.

## Objetivo do projeto

O objetivo principal é criar um assistente especializado em responder dúvidas sobre a Lei nº 15.022/2024, oferecendo respostas em português, com linguagem clara e suporte à consulta de temas legais e regulatórios.

Além disso, a solução funciona como exemplo educacional de:

- fine-tuning instrucional em domínio específico;
- uso de LoRA para reduzir custo computacional;
- preparação de datasets em formato de conversa;
- carregamento de modelo base + adapter; 
- inferência com geração de texto em português.

## Tecnologias e bibliotecas

O projeto utiliza as seguintes ferramentas e dependências principais:

- Python 3.10+
- PyTorch
- Transformers
- Datasets
- TRL (Supervised Fine-Tuning)
- PEFT (LoRA)
- Hugging Face Hub

Dependências listadas em `requirements.txt`:

- torch>=2.6.0
- transformers>=4.48.0,<5.0.0
- datasets>=3.2.0
- trl>=0.15.0
- accelerate>=1.3.0
- peft>=0.14.0

## Estrutura do projeto

```text
Fine tuning/
├── README.md
├── requirements.txt
├── lei_n_15_022_2024.json
├── output/
│   └── qwen-lei-15022-2024/
│       ├── README.md
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       ├── tokenizer.json
│       └── ...
├── src/
│   ├── configs.py
│   ├── lora_config.py
│   ├── load_dataset.py
│   ├── load_model.py
│   ├── preparar_dataset.py
│   ├── training_model.py
│   └── inferencia.py
└── .gitignore
```

## Descrição dos arquivos

### `src/configs.py`

Define os hiperparâmetros do fine-tuning, como:

- modelo base;
- número de épocas;
- tamanho do batch;
- steps de acumulação de gradiente;
- learning rate;
- tamanho máximo da sequência;
- diretório de saída do adapter.

Esses valores são essenciais para controlar a qualidade e o custo do treinamento.

### `src/lora_config.py`

Configura a adaptação LoRA aplicada ao modelo base.

Parâmetros principais:

- `r = 8`
- `lora_alpha = 16`
- `lora_dropout = 0.05`
- módulos-alvo: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- `task_type = CAUSAL_LM`

A técnica LoRA permite ajustar apenas alguns parâmetros do modelo, reduzindo retrabalho computacional e memória.

### `src/load_dataset.py`

Carrega o JSON do dataset e imprime:

- quantidade total de exemplos;
- campos presentes no primeiro item;
- primeiro prompt e primeira resposta.

Este script é útil para validar a estrutura do dataset antes do treinamento.

### `src/preparar_dataset.py`

Transforma os itens brutos do JSON em formato de conversa para o modelo:

```python
{
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

Esse formato é o padrão usado em modelos chat/instrução e é importante para o treinamento supervisionado.

### `src/load_model.py`

Carrega o tokenizer e o modelo base Qwen/Qwen2.5-0.5B-Instruct e seleciona o dispositivo disponível:

- MPS para Macs com chip Apple Silicon;
- CUDA para GPUs NVIDIA;
- CPU como fallback.

### `src/training_model.py`

É o núcleo do treinamento.

Ele:

1. carrega o arquivo JSON do dataset;
2. converte os dados para formato de chat;
3. cria um `Dataset` do Hugging Face;
4. carrega o tokenizer e o modelo base;
5. define as configurações do `SFTTrainer`;
6. aplica à configuração do LoRA;
7. executa `trainer.train()`;
8. salva o modelo e o tokenizer no diretório de saída.

O script também suporta verificação sem treinamento, com a flag:

```bash
python src/training_model.py --check-only
```

### `src/inferencia.py`

Carrega o modelo base + adapter treinado e executa uma geração textual com uma mensagem exemplo. Este script demonstra como usar o modelo ajustado para responder perguntas no contexto da lei.

Ele usa:

- `AutoTokenizer.from_pretrained(ADAPTER_PATH)`;
- `AutoModelForCausalLM.from_pretrained(MODEL_NAME)`;
- `PeftModel.from_pretrained(model_base, ADAPTER_PATH)`;
- `pipeline("text-generation", ...)`.

## Dataset

O arquivo `lei_n_15_022_2024.json` contém 896 exemplos com pares de pergunta e resposta sobre a Lei nº 15.022/2024.

Cada item possui dois campos principais:

- `prompt`: pergunta ou solicitação de resposta;
- `response`: resposta esperada em português.

O dataset cobre temas como:

- inventário nacional de substâncias químicas;
- obrigações de fabricantes e importadores;
- medidas de gerenciamento de risco;
- proteção de informações confidenciais;
- testes em animais e alternativas;
- sanções e responsabilidades;
- critérios de avaliação de risco.

Este conjunto foi estruturado para permitir treinamento de um modelo de linguagem especializado em direito ambiental e regulatório brasileiro.

## Treinamento

Antes de iniciar o treinamento, é recomendável criar um ambiente virtual:

```bash
cd "C:/caminho/para/Fine tuning"
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# ou .venv\Scripts\activate  # Windows
```

Instale as dependências:

```bash
pip install -r requirements.txt
```

Em seguida, execute:

```bash
python src/training_model.py
```

Se quiser apenas verificar que a configuração está correta sem rodar o treinamento:

```bash
python src/training_model.py --check-only
```

O modelo treinado será salvo em:

```text
output/qwen-lei-15022-2024/
```

## Inferência

Para testar o modelo ajustado, execute:

```bash
python src/inferencia.py
```

O script cria um pipeline de geração de texto e envia uma pergunta de exemplo sobre a Lei nº 15.022/2024. O resultado é impresso no terminal em português.

## Configuração do modelo

A base escolhida foi:

```text
Qwen/Qwen2.5-0.5B-Instruct
```

Essa escolha oferece um modelo leve e eficiente para fine-tuning, especialmente quando combinado com adaptação LoRA.

Os principais parâmetros de treinamento definidos em `src/configs.py` são:

- `NUM_TRAIN_EPOCHS = 3`
- `PER_DEVICE_TRAIN_BATCH_SIZE = 1`
- `GRADIENT_ACCUMULATION_STEPS = 8`
- `LEARNING_RATE = 2e-5`
- `MAX_LENGTH = 512`

Esses valores representam uma configuração de treinamento relativamente simples, porém adequada para um experimento de domínio específico em um modelo de baixa escala.

## Saída gerada pelo projeto

A pasta `output/qwen-lei-15022-2024/` contém os artefatos do adapter treinado, incluindo:

- `adapter_model.safetensors`
- `adapter_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `added_tokens.json`
- `special_tokens_map.json`
- `chat_template.jinja`
- `README.md`

Esses arquivos permitem reutilizar o modelo adaptado em aplicações de geração e integração com Transformers.

## Observações importantes

1. O projeto foi construído como um protótipo de fine-tuning especializado em legislação brasileira.
2. O modelo foi ajustado para responder em português, mas ainda pode gerar respostas incompletas ou imprecisas em contextos mais complexos.
3. A resposta final de um modelo não substitui análise jurídica profissional.
4. O projeto usa a base Qwen e depende de suporte da biblioteca Hugging Face e do ambiente local de execução.
5. Em dispositivos Apple Silicon, a execução usa MPS quando disponível; em GPUs NVIDIA, usa CUDA; em caso contrário, recorre a CPU.

## Possíveis usos

- consulta automatizada de legislação;
- assistente especializado em normas ambientais e regulatórias;
- prova de conceito para LLMs em domínio jurídico brasileiro;
- base para extensão com mais documentos legais;
- pesquisa acadêmica ou técnica em fine-tuning de modelos.

## Limitações

- o dataset é focado em perguntas e respostas específicas sobre a Lei nº 15.022/2024;
- o modelo não tem garantia de cobertura jurídica completa;
- o treinamento não substitui verificação documental oficial;
- respostas podem depender da qualidade e da quantidade de dados de treinamento;
- a inferência exige ambiente com dependências do Hugging Face corretamente instaladas.

## Licença e uso responsável

Este repositório é um projeto de pesquisa e experimentação. Antes de usar o modelo em produção, revise as licenças do modelo base e dos artefatos de terceiros, bem como o contexto de uso final e conformidade regulatória.

O uso de conteúdos legais e regulatórios deve respeitar:

- fontes oficiais;
- termos de licenciamento de módulos e dependências;
- uso responsável da IA;
- validação humana em decisões de alto impacto.

## Conclusão

O projeto demonstra uma pipeline completa de fine-tuning de um modelo de linguagem para um domínio específico: a Lei nº 15.022/2024. Ele combina dados estruturados, ajustes de arquitetura por LoRA e integração com ferramentas modernas do ecossistema Hugging Face para criar um assistente de linguagem em português, com foco em legislação ambiental e química.

Esse tipo de abordagem é útil para criar modelos especializados em diferentes áreas do conhecimento, preservando o uso de modelos base leves e reduzindo o custo de adaptação.
