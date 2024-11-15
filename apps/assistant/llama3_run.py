# from unsloth import FastLanguageModel
# from pprint import pprint
# import torch
# from peft import PeftModel
# import torch
# from peft import PeftModel
# from unsloth import FastLanguageModel

# def initialize_model():
#     max_seq_length = 256
#     dtype = None
#     load_in_4bit = True

#     # Load the base model and tokenizer
#     model, tokenizer = FastLanguageModel.from_pretrained(
#         model_name="unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
#         max_seq_length=max_seq_length,
#         dtype=dtype,
#         load_in_4bit=load_in_4bit,
#     )

#     # Load the fine-tuned model weights
#     model = PeftModel.from_pretrained(
#         model,
#         "/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/LLM_Model/ru_assistian_v3.0"
#     )

#     FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
#     return model, tokenizer

# # You're an assistant. 
# # The answer to the question you were asked using information from your dataset. 
# # The answer should be in Russian and as clear as possible for the user.
# def anser_model(process):
#     model, tokenizer = initialize_model()
#     # alpaca_prompt = """You are a sales consultant at Dmitriy Digital. 
#     # Answer the user briefly and on the topic.

#     # ### Instruction:
#     # {}

#     # ### Input:
#     # {}

#     # ### Response:
#     # {}"""
#     alpaca_prompt = """You are a professional Russian-speaking AI assistant with deep expertise in providing accurate and helpful information from your training dataset.

#     ### Role and Capabilities:
#     - Я специализированный ассистент по ботам
#     - Даю конкретные, практические ответы
#     - Использую примеры из реальной практики
#     - Предлагаю четкие шаги к решению
#     - Приоритизирую важность информации

#     ### Instruction:
#     {}

#     ### Input:
#     {}

#     ### Response Format:
#     - Прямой ответ на вопрос
#     - Дополнительные пояснения при необходимости
#     - Примеры или детали из датасета, если уместно
#     - Структурированные списки для сложной информации
#     - Заключение или следующие шаги

#     ### Response:
#     {}"""


#     instruction = "Предоставьте профессиональный ответ на вопрос пользователя с учетом контекста"  # Вместо "Ask if the client needs help..."
#     inputs = tokenizer(
#         [
#             alpaca_prompt.format(
#                 instruction, # instruction
#                 f"{process.question}", # input
#                 "", # output - leave this blank for generation!
#             )
#         ], return_tensors="pt").to("cuda")

#     # outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
#     generation_config = {
#         "do_sample": True,
#         "use_cache": True,
#         "max_new_tokens": 256
#         }
#     outputs = model.generate(**inputs, **generation_config)
#     process.answer = (tokenizer.batch_decode(outputs))
#     process.save()
    

#     return True







# from huggingface_hub import login
# import transformers
# import torch
# import csv
# login(token="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")



# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"



# def anser_model(process):
#     pipeline = transformers.pipeline(
#         "text-generation",
#         model=model_id,
#         model_kwargs={"torch_dtype": torch.bfloat16},
#         device_map="auto",

#     )

#     def load_csv_data():
#         csv_data = {}
#         with open('/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/Llama3/dataset/main_v1.0.csv', mode='r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 csv_data[row['instruction']] = row['response']
#         return csv_data

#     # Инициализация системного сообщения с информацией из CSV файла
#     csv_data = load_csv_data()

#     # Объединяем все данные CSV в строку
#     csv_info = "\n".join([f"{key}: {value}" for key, value in csv_data.items()])

#     messages = [
#         {"role": "system", "content": f"Your name is Max. You are an assistant at Dmitriy Digital. Your main task is to maintain a dialogue with the user and answer questions from this dataset: {csv_info}"},
#         {"role": "user", "content": f"{process.question}"},
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=256,
#     )
#     answer = str(outputs[0]["generated_text"][-1])
#     process.answer = answer
#     process.save()

#     return True


# from huggingface_hub import login
# import transformers
# import torch
# import csv
# from functools import lru_cache

# # Глобальные переменные для хранения инициализированных объектов

# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"



# _pipeline = None
# _csv_data = None

# def initialize_pipeline():
#     """Инициализация pipeline один раз"""
#     global _pipeline
#     if _pipeline is None:
#         login(token="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")
#         _pipeline = transformers.pipeline(
#             "text-generation",
#             model=model_id,
#             model_kwargs={
#                 "torch_dtype": torch.bfloat16,
#                 "device_map": "auto",
#                 "low_cpu_mem_usage": True,
#             }
#         )
#     return _pipeline

# @lru_cache(maxsize=1)
# def load_csv_data():
#     """Загрузка и кэширование данных CSV"""
#     global _csv_data
#     if _csv_data is None:
#         _csv_data = {}
#         with open('/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/Llama3/dataset/main_v1.0.csv', 
#                  mode='r', encoding='utf-8') as file:
#             reader = csv.DictReader(file)
#             for row in reader:
#                 _csv_data[row['instruction']] = row['response']
#         # Подготавливаем строку CSV заранее
#         _csv_data['formatted_string'] = "\n".join([f"{key}: {value}" 
#                                                  for key, value in _csv_data.items() 
#                                                  if key != 'formatted_string'])
#     return _csv_data

# def anser_model(process):
#     """Основная функция обработки запроса"""
#     try:
#         # Получаем инициализированный pipeline
#         pipeline = initialize_pipeline()
        
#         # Получаем кэшированные данные CSV
#         csv_data = load_csv_data()
        
#         # Формируем сообщения, используя предварительно отформатированную строку
#         messages = [
#             {
#                 "role": "system", 
#                 "content": f"Your name is Max. You are an assistant at Dmitriy Digital. "
#                           f"Your main task is to maintain a dialogue with the user and "
#                           f"answer questions from this dataset: {csv_data['formatted_string']}"
#             },
#             {
#                 "role": "user", 
#                 "content": process.question
#             },
#         ]

#         # Генерируем ответ
#         with torch.inference_mode():
#             outputs = pipeline(
#                 messages,
#                 max_new_tokens=64,
#                 pad_token_id=pipeline.tokenizer.eos_token_id,
#                 num_return_sequences=1,
#             )

#         # Обрабатываем результат
#         answer = str(outputs[0]["generated_text"][-1])
#         process.answer = answer
#         process.save()

#         return True

#     except Exception as e:
#         print(f"Error in anser_model: {str(e)}")
#         return False






from huggingface_hub import login
import transformers
import torch
import csv
from functools import lru_cache
import json
import os
from datetime import datetime
import re

# Глобальные переменные
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# model_id = "meta-llama/Llama-3.2-3B-Instruct"
_pipeline = None
_csv_data = None
CHAT_DIR = "/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/user_chat"  # Директория для хранения чатов


def get_chat_filename(tg_id):
    """Генерирует имя файла для чата пользователя на текущую дату"""
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(CHAT_DIR, f"{tg_id}_{today}.json")



def load_chat_history(tg_id):
    """Загружает историю чата пользователя из JSON файла или создаёт новый файл"""
    filename = get_chat_filename(tg_id)
    
    # Проверяем, существует ли файл
    if not os.path.exists(filename):
        # Если файл не существует, создаём его с пустой историей
        os.makedirs(CHAT_DIR, exist_ok=True)  # Создаём директорию, если её нет
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []  # Возвращаем пустую историю
    
    # Если файл существует, читаем его содержимое
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error reading chat file {filename}, starting new chat")
        return []





def save_chat_history(tg_id, history):
    """Сохраняет историю чата пользователя в JSON файл"""
    # Убедитесь, что директория для хранения чатов существует
    filename = get_chat_filename(tg_id)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)



def extract_user_info(message):
    """Извлекает tg_id и вопрос из сообщения"""
    pattern = r"Пользователь (\d+): (.+)"
    match = re.match(pattern, message)
    if not match:
        raise ValueError("Неверный формат сообщения")
    
    tg_id = int(match.group(1))
    question = match.group(2).strip()
    return tg_id, question



def initialize_pipeline():
    """Инициализация pipeline один раз"""
    global _pipeline
    if _pipeline is None:
        login(token="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")
        _pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "load_in_8bit": True,
            }
        )
    return _pipeline



@lru_cache(maxsize=1)
def load_csv_data():
    """Загрузка и кэширование данных CSV"""
    global _csv_data
    if _csv_data is None:
        _csv_data = {}
        with open('/home/ubuntu/machine_learning/learning/machine_learning/apps/assistant/Llama3/dataset/main_v1.0.csv', 
                 mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                _csv_data[row['instruction']] = row['response']
        _csv_data['formatted_string'] = "\n".join([f"{key}: {value}" 
                                                 for key, value in _csv_data.items() 
                                                 if key != 'formatted_string'])
    return _csv_data



def format_messages(history, question, csv_data):
    """Форматирование сообщений с учетом истории диалога"""
    messages = [
        {
            "role": "system",
            "content": f"Your name is Max. You are an assistant at Dmitriy Digital. "
                      f"Your main task is to maintain a dialogue with the user and "
                      f"answer questions from this dataset: {csv_data['formatted_string']}"
                      f"Answer all questions in no more than 128 words."
        }
    ]
    
    # Добавляем историю диалога
    messages.extend(history)
    
    # Добавляем текущий вопрос
    messages.append({"role": "user", "content": question})
    
    return messages


import ast
def anser_model(process):
    """Основная функция обработки запроса"""
    try:
        # Извлекаем tg_id и вопрос из сообщения
        tg_id, question = extract_user_info(process.question)
        
        # Загружаем историю чата
        chat_history = load_chat_history(tg_id)
        
        # Получаем инициализированный pipeline и данные CSV
        pipeline = initialize_pipeline()
        csv_data = load_csv_data()
        
        # Формируем сообщения с учетом истории диалога
        messages = format_messages(chat_history, question, csv_data)

        # Генерируем ответ
        with torch.inference_mode():
            outputs = pipeline(
                messages,
                max_new_tokens=140,
                pad_token_id=pipeline.tokenizer.eos_token_id,
                num_return_sequences=1,
            )

        # Получаем ответ модели
        answer = str(outputs[0]["generated_text"][-1])
        
        # Добавляем новые сообщения в историю
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})
        
        # Ограничиваем историю последними 10 сообщениями
        if len(chat_history) > 20:  # 10 пар вопрос-ответ
            chat_history = chat_history[-20:]
        
        # Сохраняем обновленную историю
        save_chat_history(tg_id, chat_history)
        
        # Сохраняем ответ в process

        parsed_outer = ast.literal_eval(answer)

        # Извлекаем внутреннюю строку
        inner_string = parsed_outer['content']

        # Преобразуем внутреннюю строку в словарь
        parsed_inner = ast.literal_eval(inner_string)

        # Получаем нужный текст
        answers = parsed_inner['content']

        


        process.answer = answers
        process.save()

        return True

    except Exception as e:
        print(f"Error in anser_model: {str(e)}")
        return False



def clear_user_history(tg_id: int):
    """Очистка истории диалога пользователя"""
    filename = get_chat_filename(tg_id)
    if os.path.exists(filename):
        os.remove(filename)