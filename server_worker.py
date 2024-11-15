import multiprocessing
import time
from datetime import datetime
import requests
from django.db import transaction
import os
from multiprocessing import get_context
import os
import django
from apps.assistant.llama3_run import anser_model
#     return True


from huggingface_hub import login
import transformers
import torch
import csv
from functools import lru_cache


server_ip = '62.68.146.176'
server_port = '8092'
# model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# """Инициализация pipeline один раз"""
# login(token="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")
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={
#         "torch_dtype": torch.float16,
#         "device_map": "auto",
#         "load_in_8bit": True,
#     }
# )




def process_task(process_id):
    # Установите переменную окружения с настройками вашего проекта
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'machine_learning.settings')
    django.setup()

    from apps.assistant.models import Process
    print("мы в функции process_task")
    try:
        process = Process.objects.get(id=process_id)
        print(f'----------------НАЧАЛО ГЕНЕРАЦИИ для процесса {process.id}----------------')

        
        process.process_start_time = datetime.now()
        try:
            status = anser_model(process)
            print('\nstatus',status)
        except Exception as e:
            status = False
            print('\n\n\n\ОШИБКА В ЗАПУСКЕ generation \n\n\n', e)

        process_take_time = datetime.now() - process.process_start_time

        print(f'-----------ИЗМЕНЕНИЕ В Process У ПРОЦЕССА {process.id}-------------')
        process.process_end_time = datetime.now()
        process.process_take_time = process_take_time
        process.process_ended = True
        process.save()

        print(f"ОТПРАВКА НА finish_task_status для процесса {process.id}")
        url = f"http://{server_ip}:{server_port}/finish_task_status"
        print(f'\n\n---- STATUS START {status}----\n\n')
        if status == False:
            data = {
                "task_status": 'ERROR_GENERATION',
                "task_id": process.task_id,

            }
            requests.post(url, data=data)

        else:
            data = {
                "task_status": 'COMPLETED',
                "answer": process.answer,
                "task_id": process.task_id,
            }
            requests.post(url, data=data)

        print(f'----------------ЗАВЕРШЕНИЕ ГЕНЕРАЦИИ для процесса {process.id}----------------')

    except Exception as e:
        print(f"Ошибка при обработке процесса {process_id}: {str(e)}")
        process = Process.objects.get(id=process_id)
        process.process_ended = True
        process.save()


def main():
    # Установите переменную окружения с настройками вашего проекта
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'machine_learning.settings')
    # Настройка Django
    django.setup()
    from apps.assistant.models import Process
    
    ctx = get_context('spawn')
    while True:
        try:
            unstarted_processes = Process.objects.filter(process_started=False)
            for process_obj in unstarted_processes:
                process_obj.process_started = True
                process_obj.save()
                
                print("New proccess!!!")
                # Создаем новый процесс для каждой задачи
                new_process = ctx.Process(target=process_task, args=(process_obj.id,))
                new_process.start()

        except Exception as e:
            print(f"Ошибка в главном цикле: {str(e)}")

        # print('Waiting...')
        time.sleep(1)


if __name__ == '__main__':
    main()