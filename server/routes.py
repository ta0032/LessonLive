"""
LessonLive — server/routes.py
HTTP маршруты FastAPI.
Обслуживает статические файлы, API-запросы и WebSocket соединения.
"""

import json
from pathlib import Path
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from core.session import Session
from server.socket import ConnectionManager


# ─── Пути ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.resolve()
STATIC_DIR = BASE_DIR / "static"
SETUP_DONE_FILE = BASE_DIR / ".setup_done"
ENV_FILE = BASE_DIR / ".env"


# ─── Роутер и менеджер ────────────────────────────────────────────────────────

router = APIRouter()

# Глобальный менеджер WebSocket соединений
# Создаётся один раз и используется во всех эндпоинтах
manager = ConnectionManager()


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _is_setup_complete() -> bool:
    """
    Проверяет что setup.py был успешно запущен.
    Читает .setup_done файл и проверяет что Whisper установлен.

    Returns:
        True если установка завершена и Whisper готов к работе.
    """
    if not SETUP_DONE_FILE.exists():
        return False

    try:
        with open(SETUP_DONE_FILE, "r") as f:
            status = json.load(f)
        return status.get("whisper", False)
    except Exception:
        return False


def _read_static(filename: str) -> str:
    """
    Читает содержимое статического файла из папки static/.

    Args:
        filename: Имя файла (например 'index.html').

    Returns:
        Содержимое файла как строка.

    Raises:
        HTTPException 404: Если файл не найден.
    """
    file_path = STATIC_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Файл не найден: {filename}")
    return file_path.read_text(encoding="utf-8")


# ─── Страницы ─────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def index():
    """
    Главная страница приложения.
    Если setup не завершён — перенаправляет на страницу онбординга.
    Иначе отдаёт static/index.html с интерфейсом транскрипции.
    """
    if not _is_setup_complete():
        return HTMLResponse(
            content='<meta http-equiv="refresh" content="0; url=/setup">',
            status_code=200,
        )
    return HTMLResponse(content=_read_static("index.html"))


@router.get("/setup", response_class=HTMLResponse)
async def setup_page():
    """
    Страница первоначальной настройки (онбординг).
    Отображается при первом запуске или если setup не завершён.
    Отдаёт static/setup.html.
    """
    return HTMLResponse(content=_read_static("setup.html"))


# ─── Статические файлы ────────────────────────────────────────────────────────

@router.get("/static/{filename}")
async def static_file(filename: str):
    """
    Отдаёт статические файлы (CSS, JS) из папки static/.
    Определяет Content-Type по расширению файла.

    Args:
        filename: Имя файла (например 'style.css', 'app.js').
    """
    content = _read_static(filename)

    # Определяем Content-Type
    if filename.endswith(".css"):
        media_type = "text/css"
    elif filename.endswith(".js"):
        media_type = "application/javascript"
    else:
        media_type = "text/plain"

    from fastapi.responses import Response
    return Response(content=content, media_type=media_type)


# ─── API: Настройка ───────────────────────────────────────────────────────────

@router.post("/api/setup/token")
async def save_token(request_body: dict):
    """
    Принимает HuggingFace токен из формы онбординга.
    Сохраняет в .env файл как HF_TOKEN=xxx.

    Body: {"token": "hf_xxx..."}

    Returns:
        {"success": true} или {"success": false, "error": "..."}
    """
    token = request_body.get("token", "").strip()

    if not token:
        raise HTTPException(status_code=400, detail="Токен не может быть пустым.")

    if not token.startswith("hf_"):
        raise HTTPException(status_code=400, detail="Токен должен начинаться с 'hf_'.")

    try:
        # Читаем существующий .env если есть
        lines = []
        if ENV_FILE.exists():
            with open(ENV_FILE, "r") as f:
                lines = [l for l in f.readlines() if not l.startswith("HF_TOKEN=")]

        # Добавляем токен
        lines.append(f"HF_TOKEN={token}\n")

        with open(ENV_FILE, "w") as f:
            f.writelines(lines)

        return JSONResponse({"success": True})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/api/setup/status")
async def setup_status():
    """
    Возвращает статус установки из .setup_done файла.
    Используется страницей онбординга для отображения прогресса.

    Returns:
        {"whisper": bool, "pyannote": bool, "vbcable": bool}
    """
    if not SETUP_DONE_FILE.exists():
        return JSONResponse({"whisper": False, "pyannote": False, "vbcable": False})

    try:
        with open(SETUP_DONE_FILE, "r") as f:
            return JSONResponse(json.load(f))
    except Exception:
        return JSONResponse({"whisper": False, "pyannote": False, "vbcable": False})


# ─── API: Сессии ──────────────────────────────────────────────────────────────

@router.get("/api/sessions")
async def get_sessions():
    """
    Возвращает список всех сохранённых файлов транскрипций.
    Отсортирован по дате — новые первые.

    Returns:
        [{"filename": "...", "date": "...", "size_kb": ...}, ...]
    """
    sessions = Session.get_sessions_list()
    return JSONResponse(sessions)


@router.get("/api/sessions/{filename}")
async def get_session_file(filename: str):
    """
    Возвращает содержимое конкретного файла транскрипции.
    Защищён от path traversal атак.

    Args:
        filename: Имя файла (например '2024-03-14_22-30_lesson.txt').

    Returns:
        Текстовое содержимое файла.

    Raises:
        HTTPException 404: Если файл не найден.
        HTTPException 403: Если путь выходит за пределы sessions/.
    """
    try:
        content = Session.read_session_file(filename)
        return PlainTextResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Файл не найден: {filename}")
    except PermissionError:
        raise HTTPException(status_code=403, detail="Доступ запрещён.")


# ─── API: Спикеры ─────────────────────────────────────────────────────────────

@router.post("/api/speaker-names")
async def set_speaker_names(request_body: dict, app_state=None):
    """
    Устанавливает имена спикеров.
    Вызывается из интерфейса когда пользователь вводит имена.

    Body: {"speaker_0": "Антуан", "speaker_1": "Мария"}

    Returns:
        {"success": true, "speakers": [{"id": ..., "name": ...}, ...]}
    """
    # app_state передаётся через dependency injection (настраивается в app.py)
    # Здесь используем manager который имеет доступ к app_state
    await manager.handle_message(
        websocket=None,
        data=json.dumps({"action": "set_names", **request_body}),
        app_state=app_state,
    )
    return JSONResponse({"success": True})


@router.get("/api/status")
async def get_app_status():
    """
    Возвращает текущее состояние приложения.
    Используется браузером при загрузке страницы для синхронизации UI.

    Returns:
        {
            "is_recording": bool,
            "model_loaded": bool,
            "pyannote_loaded": bool,
            "phrase_count": int,
            "session_file": str | null,
            "vbcable_available": bool
        }
    """
    # Статус формируется на основе данных менеджера
    # Полные данные о моделях добавляются в app.py через middleware
    return JSONResponse(manager._build_status())


# ─── WebSocket ────────────────────────────────────────────────────────────────

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket эндпоинт — основной канал связи с браузером.
    При подключении: отправляет статус и историю сессии.
    Слушает входящие команды: start, stop, set_names.
    При отключении: корректно удаляет соединение.

    Протокол сообщений от клиента:
        {"action": "start"}
        {"action": "stop"}
        {"action": "set_names", "speaker_0": "...", "speaker_1": "..."}

    Протокол сообщений от сервера:
        {"type": "status", "is_recording": bool, ...}
        {"type": "history", "phrases": [...]}
        {"type": "phrase", "speaker_id": "...", "text": "...", ...}
        {"type": "error", "message": "..."}
    """
    await manager.connect(websocket)

    try:
        while True:
            # Ждём команду от клиента
            data = await websocket.receive_text()

            # Получаем app_state из app модуля (избегаем циклический импорт)
            from app import app_state
            await manager.handle_message(websocket, data, app_state)

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"  [WS] Неожиданная ошибка: {e}")
        await manager.disconnect(websocket)
