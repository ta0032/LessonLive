"""
LessonLive — app.py
Точка входа приложения. Только сервер — без открытия браузера.
Открытием браузера управляет launch.pyw.
"""

import os
import sys
import json
import asyncio
import threading
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Принудительная буферизация для subprocess pipe
import sys
sys.stdout.reconfigure(line_buffering=True)

# Фикс для Windows: ProactorEventLoop конфликтует с PortAudio WASAPI
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BASE_DIR = Path(__file__).parent.resolve()
SETUP_DONE_FILE = BASE_DIR / ".setup_done"
ENV_FILE = BASE_DIR / ".env"

def load_env():
    if ENV_FILE.exists():
        load_dotenv(ENV_FILE)

def check_setup_complete() -> bool:
    if not SETUP_DONE_FILE.exists():
        return False
    try:
        with open(SETUP_DONE_FILE, "r") as f:
            status = json.load(f)
        return status.get("whisper", False)
    except Exception:
        return False

from state import app_state, AppState

def load_models():
    from core.audio import AudioCapture
    from core.transcribe import Transcriber
    from core.diarize import Diarizer

    model_size = _get_model_size()
    print(f"  Загружаю модели... (модель: {model_size})", flush=True)

    app_state.audio_capture = AudioCapture(chunk_duration=3.0, sample_rate=16000)

    try:
        app_state.transcriber = Transcriber(model_size=model_size).load()
        app_state.model_loaded = True
        print("  ✓ Whisper загружен.", flush=True)
    except Exception as e:
        print(f"  ✗ Ошибка загрузки Whisper: {e}", flush=True)

    token = os.environ.get("HF_TOKEN")
    try:
        app_state.diarizer = Diarizer(token=token).load()
        app_state.pyannote_loaded = True
        print("  ✓ pyannote загружен.", flush=True)
    except Exception as e:
        print(f"  ⚠ pyannote не загружен: {e}", flush=True)

    print("  Все модели загружены. Приложение готово.", flush=True)

    asyncio.run_coroutine_threadsafe(
        manager.broadcast_status(),
        loop=app_state._event_loop,
    )

def _get_model_size() -> str:
    try:
        with open(SETUP_DONE_FILE, "r") as f:
            status = json.load(f)
        return status.get("model_size", "small")
    except Exception:
        return "small"

def create_app() -> FastAPI:
    from server.routes import router

    application = FastAPI(
        title="LessonLive",
        description="Real-time транскрипция для онлайн-занятий",
        version="1.0.0",
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router)

    @application.on_event("startup")
    async def on_startup():
        app_state._event_loop = asyncio.get_event_loop()
        threading.Thread(target=load_models, daemon=True).start()

    return application

def start_server():
    """Запускает uvicorn. Без открытия браузера — это делает launch.pyw."""
    import uvicorn
    print("  Сервер запускается на http://localhost:8000")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        log_level="warning",
    )

_event_loop: asyncio.AbstractEventLoop = None
app = create_app()

from server.routes import manager
manager._app_state = app_state

if __name__ == "__main__":
    load_env()

    if not check_setup_complete():
        print("  Установка не завершена. Запустите setup.py")
        sys.exit(1)

    start_server()
