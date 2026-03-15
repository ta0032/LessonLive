"""
LessonLive — app.py
Точка входа приложения.
Инициализирует все компоненты, связывает pipeline и запускает сервер.
"""

import os
import sys
import json
import asyncio
import threading
import webbrowser
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Фикс для Windows: ProactorEventLoop конфликтует с PortAudio WASAPI
# Переключаемся на SelectorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ─── Пути ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()
SETUP_DONE_FILE = BASE_DIR / ".setup_done"
ENV_FILE = BASE_DIR / ".env"


# ─── Загрузка окружения ───────────────────────────────────────────────────────

def load_env():
    """
    Загружает переменные окружения из .env файла.
    Делает HF_TOKEN доступным для pyannote при загрузке модели.
    """
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)


# ─── Проверка установки ───────────────────────────────────────────────────────

def check_setup_complete() -> bool:
    """
    Проверяет что setup.py был успешно запущен.
    Читает .setup_done и проверяет что минимум Whisper установлен.

    Returns:
        True если можно запускать приложение.
    """
    if not SETUP_DONE_FILE.exists():
        return False
    try:
        with open(SETUP_DONE_FILE, "r") as f:
            status = json.load(f)
        return status.get("whisper", False)
    except Exception:
        return False


# Глобальный экземпляр состояния (импортируется из routes.py и socket.py)
from state import app_state, AppState

# ─── Инициализация моделей ────────────────────────────────────────────────────

def load_models():
    """
    Загружает все AI-модели при старте приложения.
    Выполняется в отдельном потоке чтобы не блокировать запуск сервера.
    Модели загружаются последовательно: сначала Whisper, потом pyannote.
    """
    from core.audio import AudioCapture
    from core.transcribe import Transcriber
    from core.diarize import Diarizer

    # Определяем размер модели из .setup_done
    model_size = _get_model_size()
    print(f"\n  Загружаю модели... (модель: {model_size})")

    # Инициализируем AudioCapture
    app_state.audio_capture = AudioCapture(chunk_duration=3.0, sample_rate=16000)

    # Загружаем Whisper
    try:
        app_state.transcriber = Transcriber(model_size=model_size).load()
        app_state.model_loaded = True
        print("  ✓ Whisper загружен.")
    except Exception as e:
        print(f"  ✗ Ошибка загрузки Whisper: {e}")

    # Загружаем pyannote (опционально)
    token = os.environ.get("HF_TOKEN")
    try:
        app_state.diarizer = Diarizer(token=token).load()
        app_state.pyannote_loaded = True
        print("  ✓ pyannote загружен.")
    except Exception as e:
        print(f"  ⚠ pyannote не загружен (разделение спикеров недоступно): {e}")

    print("  Все модели загружены. Приложение готово.\n")

    # Уведомляем клиентов что статус изменился
    asyncio.run_coroutine_threadsafe(
        manager.broadcast_status(),
        loop=app_state._event_loop,
    )


def _get_model_size() -> str:
    """
    Читает выбранный размер модели из .setup_done.
    По умолчанию возвращает 'medium'.
    """
    try:
        with open(SETUP_DONE_FILE, "r") as f:
            status = json.load(f)
        return status.get("model_size", "medium")
    except Exception:
        return "medium"


# ─── Открытие браузера ────────────────────────────────────────────────────────

def open_browser():
    """
    Открывает браузер на localhost:8000 через 1.5 секунды после старта.
    Задержка нужна чтобы uvicorn успел подняться.
    Запускается в отдельном потоке.
    """
    import time
    time.sleep(1.5)
    webbrowser.open("http://localhost:8000")
    print("  Браузер открыт: http://localhost:8000")


# ─── FastAPI приложение ───────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Создаёт и настраивает FastAPI приложение.
    Подключает роутер с маршрутами и CORS middleware.

    Returns:
        Настроенный экземпляр FastAPI.
    """
    from server.routes import router

    application = FastAPI(
        title="LessonLive",
        description="Real-time транскрипция для онлайн-занятий",
        version="1.0.0",
    )

    # CORS — разрешаем запросы только с localhost
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Подключаем все маршруты
    application.include_router(router)

    # При старте сервера — загружаем модели в фоне
    @application.on_event("startup")
    async def on_startup():
        app_state._event_loop = asyncio.get_event_loop()
        threading.Thread(target=load_models, daemon=True).start()

    return application


# ─── Запуск ───────────────────────────────────────────────────────────────────

def start_server():
    """
    Запускает uvicorn сервер на localhost:8000.
    Одновременно в отдельном потоке открывает браузер.
    """
    import uvicorn

    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║              LessonLive v1.0                     ║")
    print("║      Не закрывайте это окно во время работы!     ║")
    print("╚══════════════════════════════════════════════════╝")
    print()
    print("  Сервер запускается на http://localhost:8000")

    # Открываем браузер в отдельном потоке
    threading.Thread(target=open_browser, daemon=True).start()

    # Запускаем сервер
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",    # Только важные логи — не засоряем консоль
    )


# ─── Глобальные переменные ────────────────────────────────────────────────────

# Event loop для вызова async функций из синхронных потоков
_event_loop: asyncio.AbstractEventLoop = None

# Создаём приложение
app = create_app()

# Импортируем manager здесь чтобы избежать циклических импортов
from server.routes import manager
manager._app_state = app_state


# ─── Точка входа ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_env()

    if not check_setup_complete():
        print()
        print("  ✗ Установка не завершена.")
        print("  Запустите setup.py для первоначальной настройки:")
        print("  python setup.py")
        print()
        sys.exit(1)

    start_server()
