"""
LessonLive — state.py
Глобальное состояние приложения — синглтон.
Единственный экземпляр app_state импортируется везде где нужен.
"""
import asyncio
import numpy as np


class AppState:
    def __init__(self):
        self.audio_capture = None
        self.transcriber = None
        self.diarizer = None
        self.session = None
        self.model_loaded = False
        self.pyannote_loaded = False
        self._event_loop = None

    def on_audio_chunk(self, audio_chunk: np.ndarray, source: str = "microphone"):
        if not self.session or not self.session.is_active:
            return
        if not self.transcriber or not self.model_loaded:
            return

        try:
            result = self.transcriber.transcribe(audio_chunk)
            if result.is_empty:
                return

            if source == "microphone":
                speaker_id = "speaker_0"
                speaker_name = self.diarizer.get_speaker_name("speaker_0") if self.diarizer else "Преподаватель"
            elif source == "loopback":
                # Временно отключаем pyannote для loopback — слишком тяжело
                speaker_id = "speaker_1"
                speaker_name = self.diarizer.get_speaker_name("speaker_1") if self.diarizer else "Ученик"
            else:
                speaker_id = "unknown"
                speaker_name = "Неизвестно"

            phrase = self.session.add_phrase(
                speaker_id=speaker_id,
                speaker_name=speaker_name,
                text=result.text,
                language=result.language,
            )

            if self._event_loop:
                from server.routes import manager
                asyncio.run_coroutine_threadsafe(
                    manager.broadcast(phrase.to_dict()),
                    loop=self._event_loop,
                )

        except Exception as e:
            print(f"  [Pipeline] Ошибка обработки чанка: {e}")


# Единственный экземпляр — создаётся один раз при импорте
app_state = AppState()