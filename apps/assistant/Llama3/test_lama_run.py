
from huggingface_hub import login
import transformers
import torch
import csv
login(token="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")

from pipline_run import run



model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)


def load_csv_data():
    csv_data = {}
    with open('/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/Llama3/dataset/main_v1.0.csv', mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            csv_data[row['instruction']] = row['response']
    return csv_data

# Инициализация системного сообщения с информацией из CSV файла
csv_data = load_csv_data()

# Объединяем все данные CSV в строку
csv_info = "\n".join([f"{key}: {value}" for key, value in csv_data.items()])

messages = [
    {"role": "system", "content": f"Your name is Max. You are an assistant at Dmitriy Digital. Your main task is to maintain a dialogue with the user and answer questions from this dataset: {csv_info}"},
    {"role": "user", "content": f"{question}"},
]

outputs = pipeline(
    messages,
    max_new_tokens=256,
)
answer = int(outputs[0]["generated_text"][-1])
print(answer['content'])
