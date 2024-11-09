# Используем официальный образ Python как базовый
FROM python:3.10-slim
# Устанавливаем рабочую директорию
WORKDIR /app
# Копируем файлы с зависимостями в рабочую директорию
COPY requirements.txt .
# Устанавливаем зависимости
RUN pip install --no-cache-dir -r requirements.txt
# Копируем код приложения в рабочую директорию
COPY . .
# Устанавливаем переменные окружения
ENV PYTHONUNBUFFERED=1
# Открываем порты для приложения
EXPOSE 8000
# Команда для запуска FastAPI
CMD ["uvicorn", "app.app:app", "--host", "0.0.0.0", "--port", "8000"]