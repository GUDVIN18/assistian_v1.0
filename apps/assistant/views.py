from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json
from apps.assistant.models import ServerConfig, Process
from datetime import datetime
import requests
import traceback
import multiprocessing
import os


@csrf_exempt  
def get_data(request):
    if request.method == "POST":
        server_auth_token = request.POST.get("server_auth_token")
        try:
            if ServerConfig.objects.first().auth_token == server_auth_token:
                server_max_process = request.POST.get("server_max_process")
                process_backend_id = request.POST.get("process_backend_id") #
                task_id = request.POST.get("task_id")
                question = request.POST.get("question")
                server_address = request.POST.get("server_address")
                server_port = request.POST.get("server_port")
                

                


                process_start = Process.objects.filter(process_ended=False)
                print(f'--------ЗАПУЩЕННЫЕ ПРОЦЕССЫ {process_start}----------')
                
                if not process_start.exists() or len(process_start) < int(server_max_process):
                    

                    process_create = Process.objects.create(
                        process_backend_id = process_backend_id,
                        question = question,
                        process_started = False,
                        maximum_number_processes = server_max_process,
                    )

                    print(f'--------ПРОЦЕСС СОЗДАН {process_create.id}----------')
                    
                    if process_create:
                        try:
                            url = f"http://{server_address}:8092/get_task_status"
                            data = {
                                "task_status": 'ACCEPTED',
                                "id": task_id,
                            }
                            requests.post(url, data=data)
                            print(url)



                            update_process = Process.objects.get(id=process_create.id)
                            update_process.task_id = task_id
                            update_process.save()
                            return HttpResponse(json.dumps({f"success": f"{ServerConfig.objects.first().config_title} WORKER START"}), status=200)

    
                        except Exception as e:
                            process = Process.objects.get(id=process_create.id)
                            if process:
                                process.process_error = str(e)
                                process.process_error_traceback = traceback.format_exc()
                                process.save() 
                            return HttpResponse(json.dumps({f"error": f"Exception as"}), status=200)
                    else:
                        print(f'--------ПРОЦЕСС НЕ СОЗДАН----------')
                        return HttpResponse(f"Not Created {ServerConfig.objects.first().config_title}")
                else:
                    return HttpResponse(f"MAX Proccess {ServerConfig.objects.first().config_title}")
            else:
                return HttpResponse(f"Invalid Token {ServerConfig.objects.first().config_title}")
        except Exception as e:
            print(f'Error {ServerConfig.objects.first().config_title}', e)
            return HttpResponse(json.dumps({"error": f"Access error > {e}"}), status=500)
    else:
        print(request.method)
        return HttpResponse(f"Ошибка запроса {ServerConfig.objects.first().config_title}")