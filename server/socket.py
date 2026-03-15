"""
LessonLive — server/socket.py
WebSocket менеджер для мгновенной доставки транскрипции в браузер.
Управляет подключениями клиентов и рассылкой сообщений.
"""

import json
import asyncio
from state import app_state
from typing import Optional
from fastapi import WebSocket, WebSocketDisconnect


# ─── Класс ConnectionManager ──────────────────────────────────────────────────

class ConnectionManager:
    """
    Управляет всеми активными WebSocket соединениями.
    Обеспечивает broadcast новых фраз всем подключённым клиентам.
    При подключении нового клиента отправляет ему всю историю сессии.

    Пример использования (внутри FastAPI):
        manager = ConnectionManager()

        @app.websocket("/ws")
        async def ws_endpoint(websocket: WebSocket):
            await manager.connect(websocket)
    """

    def __init__(self):
        """
        Инициализирует пустой список активных соединений.
        Ссылка на текущую сессию устанавливается извне через set_session().
        """
        self.active_connections: list[WebSocket] = []
        self._session = None       # Текущая Session (устанавливается из app.py)
        self._is_recording = False # Флаг активной записи

    # ─── Управление соединениями ──────────────────────────────────────────────

    async def connect(self, websocket: WebSocket):
        """
        Принимает новое WebSocket соединение.
        Добавляет в active_connections.
        Отправляет клиенту текущий статус и историю сессии
        чтобы он увидел уже накопленный текст.

        Args:
            websocket: Объект WebSocket соединения от FastAPI.
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"  [WS] Клиент подключён. Всего: {len(self.active_connections)}")

        # Отправляем текущий статус
        status = self._build_status()
        print(f"  [WS] Отправляю статус: {status}")
        await self._send_json(websocket, status)

        # Отправляем историю текущей сессии если она есть
        if self._session and self._session.is_active:
            history = self._session.get_history()
            if history:
                await self._send_json(websocket, {
                    "type": "history",
                    "phrases": history,
                })

    async def disconnect(self, websocket: WebSocket):
        """
        Удаляет соединение из active_connections при отключении клиента.
        Вызывается при закрытии вкладки или обрыве соединения.

        Args:
            websocket: Объект WebSocket который отключился.
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"  [WS] Клиент отключён. Осталось: {len(self.active_connections)}")

    # ─── Рассылка сообщений ───────────────────────────────────────────────────

    async def broadcast(self, message: dict):
        """
        Отправляет JSON сообщение всем подключённым клиентам.
        Если клиент отключился в процессе — удаляет его из списка.

        Args:
            message: Словарь который будет сериализован в JSON.
                     Ожидаемая структура для фразы:
                     {
                         "type": "phrase",
                         "index": 0,
                         "speaker_id": "speaker_0",
                         "speaker_name": "Антуан",
                         "text": "Bonjour!",
                         "language": "fr",
                         "timestamp": "22:31:05"
                     }
        """
        disconnected = []

        for websocket in self.active_connections:
            try:
                await self._send_json(websocket, message)
            except Exception:
                disconnected.append(websocket)

        # Убираем отвалившихся клиентов
        for websocket in disconnected:
            await self.disconnect(websocket)

    async def broadcast_status(self):
        """
        Рассылает всем клиентам текущий статус приложения.
        Вызывается при старте/остановке записи и смене имён спикеров.
        """
        status = self._build_status()
        print(f"  [WS] broadcast_status: {status}")
        await self.broadcast(status)

    # ─── Обработка команд от клиента ─────────────────────────────────────────

    async def handle_message(self, websocket: WebSocket, data: str, app_state):
        """
        Обрабатывает входящее сообщение от браузера.
        Браузер может отправлять команды управления записью и именами спикеров.

        Args:
            websocket:  WebSocket отправителя.
            data:       Сырая строка JSON сообщения.
            app_state:  Объект AppState из app.py с доступом к pipeline.

        Поддерживаемые команды:
            {"action": "start"}                          — начать запись
            {"action": "stop"}                           — остановить запись
            {"action": "set_names",
             "speaker_0": "Имя", "speaker_1": "Имя"}    — назначить имена спикеров
        """
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            await self._send_json(websocket, {"type": "error", "message": "Неверный формат JSON."})
            return

        action = message.get("action")

        if action == "start":
            await self._handle_start(app_state)

        elif action == "stop":
            await self._handle_stop(app_state)

        elif action == "set_names":
            await self._handle_set_names(message, app_state)

        else:
            await self._send_json(websocket, {
                "type": "error",
                "message": f"Неизвестная команда: {action}"
            })

    # ─── Внутренние обработчики команд ───────────────────────────────────────

    async def _handle_start(self, app_state):
        if self._is_recording:
            return

        # Используем _app_state если доступен (правильная ссылка)
        state = app_state if hasattr(self, '_app_state') and app_state else app_state

        if not app_state.audio_capture:
            await self.broadcast({"type": "error", "message": "Аудио захват не инициализирован. Подождите загрузки моделей."})
            return

        try:
            from core.session import Session
            self._session = Session()
            app_state.session = self._session
            self.set_session(self._session)
            self._session.start()

            app_state.audio_capture.start(callback=app_state.on_audio_chunk)

            self._is_recording = True
            print("  [WS] Запись запущена.")

        except Exception as e:
            print(f"  [WS] Ошибка запуска записи: {e}")
            await self.broadcast({"type": "error", "message": str(e)})
            return

        await self.broadcast_status()

    async def _handle_stop(self, app_state):
        if not self._is_recording:
            return

        state = app_state if hasattr(self, '_app_state') and app_state else app_state

        try:
            app_state.audio_capture.stop()
            if self._session:
                self._session.stop()
            self._is_recording = False
            print("  [WS] Запись остановлена.")
        except Exception as e:
            print(f"  [WS] Ошибка остановки записи: {e}")

        await self.broadcast_status()

    async def _handle_set_names(self, message: dict, app_state):
        """
        Устанавливает имена спикеров через diarizer.
        Рассылает обновлённый статус всем клиентам.

        Args:
            message:   Словарь с полями speaker_0 и/или speaker_1.
            app_state: AppState из app.py.
        """
        diarizer = app_state.diarizer
        if not diarizer:
            return

        for speaker_id in ("speaker_0", "speaker_1"):
            name = message.get(speaker_id, "").strip()
            if name:
                diarizer.set_speaker_name(speaker_id, name)

        await self.broadcast_status()

    # ─── Вспомогательные методы ───────────────────────────────────────────────

    def set_session(self, session):
        """
        Устанавливает ссылку на текущую сессию.
        Вызывается из app.py при создании новой сессии.

        Args:
            session: Объект Session или None.
        """
        self._session = session

    def _build_status(self) -> dict:
        from state import app_state
        return {
            "type": "status",
            "is_recording": self._is_recording,
            "phrase_count": self._session.get_phrase_count() if self._session else 0,
            "session_file": self._session.file_path.name if self._session else None,
            "model_loaded": app_state.model_loaded,
            "pyannote_loaded": app_state.pyannote_loaded,
        }

    async def _send_status(self, websocket: WebSocket):
        """Отправляет статус конкретному клиенту."""
        await self._send_json(websocket, self._build_status())

    async def _send_json(self, websocket: WebSocket, data: dict):
        """
        Отправляет JSON словарь конкретному WebSocket клиенту.

        Args:
            websocket: Целевое соединение.
            data:      Словарь для сериализации.
        """
        await websocket.send_text(json.dumps(data, ensure_ascii=False))
