from unsloth import FastLanguageModel
import torch
import wandb
wb_token = '74516394b0ceffad946883c8457556d6ff7687d0' # ключ c сайта Weights & Biases https://wandb.ai/site
wandb.login(key=wb_token)

#https://github.com/a-milenkin/LLM_practical_course/blob/main/notebooks/M5_2_FineTuning.ipynb


max_seq_length = 256 # Choose any! We auto support RoPE Scaling internally!
dtype = None          # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True   # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit, # Применяем QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)


# alpaca_prompt = """You're an assistant. 
# The answer to the question you were asked using information from your dataset. 
# The answer should be in Russian and as clear as possible for the user..

# ### Instruction:
# {}

# ### Input:
# {}

# ### Response:
# {}"""


alpaca_prompt = """You are a professional Russian-speaking AI assistant with deep expertise in providing accurate and helpful information from your training dataset.

### Role 

### Instruction:
{}

### Input:
{}

### Response Format:
- Прямой ответ на вопрос
- Дополнительные пояснения при необходимости
- Примеры или детали из датасета, если уместно
- Структурированные списки для сложной информации
- Заключение или следующие шаги

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# Функция преобразования полей датасета в alpaca prompt
def formatting_prompts_func(examples):
    instructions = examples["Instruction"]
    inputs       = examples["Input"]
    outputs      = examples["Response"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }



from pprint import pprint
from datasets import load_dataset
import pandas as pd
from datasets import Dataset
# Скачиваем подготовленный датасет с HuggingFace 
# dataset = load_dataset("Ivanich/datafeeling_posts", split = "train")
dataset = pd.read_csv('/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/dataset/chatbot_client_questions_responses.csv')
dataset = Dataset.from_pandas(dataset)

# Преобразуем в alpaca_prompt с помощью нашей функции и метода map.
dataset = dataset.map(formatting_prompts_func, batched = True,)
print("\n\n\ndataset",dataset)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 2,
        warmup_steps = 20, #10,
        #num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 1000,
        learning_rate = 3e-4,
        fp16=False,
        bf16=True,
        logging_steps = 1,
        optim="paged_adamw_32bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to="wandb", # Если используете Weights & Biases
    ),
)


#@title Show current memory stats
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")



# Запускаем тренировку!
trainer_stats = trainer.train()

#@title Show final memory and time stats
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


model.save_pretrained("./LLM_Model/ru_assistian_v4.0") # Local saving



#Использование
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    alpaca_prompt.format(
        "Ask if the client needs help with formulating specific chatbot functionalities", # instruction
        "Хочу чат-бота для моего бизнеса.", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 128, use_cache = True)
pprint(tokenizer.batch_decode(outputs))