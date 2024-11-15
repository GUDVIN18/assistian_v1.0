from django.contrib import admin
from .models import ServerConfig, Process

@admin.register(ServerConfig)
class ServerConfigPartAdmin(admin.ModelAdmin):
    fields = [
        "config_title",
        "auth_token",
    ]
    list_filter = (
        "config_title",
        "auth_token",
    )
    search_fields = (
        "config_title",
        "auth_token",
    )



@admin.register(Process)
class ProcessPartAdmin(admin.ModelAdmin):
    fields = [
        "process_started",
        "question",
        "answer",
        "process_start_time",
        "process_end_time",
        "process_take_time",
        "process_ended",
        "process_error",
        "process_error_traceback",
        "process_backend_id",
        "maximum_number_processes", 
        "task_id",

    ]
    list_display = (
        "id",
        "process_started",
        "process_ended",
        "process_take_time",
        "process_start_time",
    )
    # readonly_fields = (
    #     "",
    #     "",
    # )
    list_filter = (
        "process_backend_id",
        "process_started",
    )
    search_fields = (
        "process_backend_id",
        "process_started",
    )