from django.db import models


class ServerConfig(models.Model):
    config_title = models.CharField(max_length=100, verbose_name="Название сервера", blank=True, null=True)
    auth_token = models.CharField(max_length=100, verbose_name="Токен", blank=True, null=True)

    def __str__(self):
        return self.config_title

    class Meta:
        verbose_name = "Настройка сервера"
        verbose_name_plural = "Настройка сервера"







class Process(models.Model):
    process_start_time = models.DateTimeField(null=True, blank=True, help_text="Время старта процесса", verbose_name='Старт')
    process_end_time = models.DateTimeField(null=True, blank=True, help_text="Время окончания процесса (если процесс завершен)", verbose_name='Завершение')
    process_take_time = models.DurationField(null=True, blank=True, help_text="Сколько всего времени заняло исполнение процесса (если процесс завершен)", verbose_name='Время выполнения')
    process_ended = models.BooleanField(default=False, help_text="True или False. Если True - значит процесс завершен, если False - значит процесс все еще в работе", verbose_name='Статус')
    process_error = models.TextField(null=True, blank=True, help_text="Ошибка при выполнении процесса (если возникла)", verbose_name='Ошибка')
    process_error_traceback = models.TextField(null=True, blank=True, help_text="Трейсбэк ошибки (если возникла ошибка)", verbose_name='Трейсбэк')
    process_backend_id = models.CharField(max_length=254, unique=True, help_text="Уникальный идентификатор процесса, созданный на бэкенде", verbose_name = 'id процесса')
    process_started = models.BooleanField(default='', help_text="Если процесс запущен - True, если нет - False.", verbose_name='Запуск процесса')
    maximum_number_processes = models.CharField(max_length=10, help_text="Максимальное кол-во процессов", verbose_name='Кол-во процессов', null=True, blank=True)

    question = models.TextField(help_text='Вопрос пользовтеля')
    answer = models.TextField(help_text='Ответ модели', blank=True, null=True)
    task_id = models.IntegerField(verbose_name="task_id", help_text="Номер task", null=True, blank=True)

    def __str__(self):
        return f"Process {self.process_backend_id}"
    
    class Meta:
        verbose_name = "Процесс"
        verbose_name_plural = "Процессы"


