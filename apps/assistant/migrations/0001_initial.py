# Generated by Django 5.1.3 on 2024-11-08 10:03

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='QuestionsAnswer',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField(help_text='Вопрос')),
                ('answer', models.TextField(blank=True, help_text='Ответ', null=True)),
            ],
            options={
                'verbose_name': 'Question and Answer',
                'verbose_name_plural': 'Questions and Answers',
            },
        ),
        migrations.CreateModel(
            name='Task',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('process_start_time', models.DateTimeField(blank=True, help_text='Время старта процесса', null=True, verbose_name='Старт')),
                ('process_end_time', models.DateTimeField(blank=True, help_text='Время окончания процесса (если процесс завершен)', null=True, verbose_name='Завершение')),
                ('process_take_time', models.DurationField(blank=True, help_text='Сколько всего времени заняло исполнение процесса (если процесс завершен)', null=True, verbose_name='Время выполнения')),
                ('process_error', models.TextField(blank=True, help_text='Ошибка при выполнении процесса (если возникла)', null=True, verbose_name='Ошибка')),
                ('process_error_traceback', models.TextField(blank=True, help_text='Трейсбэк ошибки (если возникла ошибка)', null=True, verbose_name='Трейсбэк')),
                ('task_id_request', models.IntegerField()),
                ('accept', models.BooleanField(default=False)),
                ('completed', models.BooleanField(default=False)),
                ('question', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='question_tasks', to='assistant.questionsanswer')),
            ],
            options={
                'verbose_name': 'Task',
                'verbose_name_plural': 'Tasks',
                'db_table': 'assistant_task',
            },
        ),
    ]
