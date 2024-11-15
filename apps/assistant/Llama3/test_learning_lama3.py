import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os

# Инициализация модели
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
model = transformers.LlamaForCausalLM.from_pretrained(model_id)
tokenizer = transformers.LlamaTokenizer.from_pretrained(model_id)

# Загрузка датасета из JSON
class JsonDataset(Dataset):
    def __init__(self, json_file):
        self.data = []
        with open(json_file, "r") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

dataset = JsonDataset("/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/Llama3/dataset/main_v1.0.jsonl")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Настройка параметров обучения
learning_rate = 2e-5
num_epochs = 3

# Дообучение модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = transformers.AdamW(model.parameters(), lr=learning_rate)
scheduler = transformers.get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=len(dataloader) * num_epochs
)

model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = tokenizer(batch["text"], return_tensors="pt", padding=True).input_ids.to(device)
        output = model(input_ids, labels=input_ids)
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

# Сохранение обученной модели
model.save_pretrained("./asistian_bot_LLM")