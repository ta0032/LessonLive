"""
LessonLive — core/transcribe.py
Транскрипция аудио-чанков в текст с помощью faster-whisper.
Автоматически определяет язык (французский, русский, английский).
"""

import os
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ─── Пути ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.resolve()
WHISPER_DIR = BASE_DIR / "models" / "whisper"


# ─── Результат транскрипции ───────────────────────────────────────────────────

@dataclass
class TranscriptionResult:
    """
    Результат транскрипции одного аудио-чанка.

    Атрибуты:
        text:       Распознанный текст. Пустая строка если речь не обнаружена.
        language:   Код языка: 'fr', 'ru', 'en' или другой ISO 639-1 код.
        confidence: Уверенность модели от 0.0 до 1.0.
        is_empty:   True если текст пустой или содержит только шум.
    """
    text: str
    language: str
    confidence: float

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


# ─── Класс Transcriber ────────────────────────────────────────────────────────

class Transcriber:
    """
    Загружает модель faster-whisper и транскрибирует аудио-чанки в текст.
    Поддерживает французский, русский и английский языки одновременно.

    Пример использования:
        transcriber = Transcriber(model_size="medium").load()
        result = transcriber.transcribe(audio_chunk)
        print(result.text, result.language)
    """

    # Языки которые мы ожидаем — ограничение ускоряет определение языка
    TARGET_LANGUAGES = ["fr", "ru", "en"]

    def __init__(
        self,
        model_size: str = "medium",
        device: str = "auto",
    ):
        """
        Инициализирует параметры транскрипции.

        Args:
            model_size: Размер модели Whisper: 'small', 'medium', 'large-v3'.
                        Чем больше модель — тем лучше качество и медленнее работа.
            device:     Устройство для запуска: 'auto', 'cuda', 'cpu'.
                        'auto' — автоматически выбирает GPU если доступен.
        """
        self.model_size = model_size
        self.device = self._resolve_device(device)
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        self._model = None
        self.is_loaded = False

        print(f"  [Transcriber] Параметры: model={model_size}, device={self.device}, compute={self.compute_type}")

    # ─── Публичные методы ─────────────────────────────────────────────────────

    def load(self) -> "Transcriber":
        """
        Загружает модель faster-whisper из локальной папки models/whisper/.
        Вызывается один раз при старте приложения.
        Возвращает self для цепочки вызовов: Transcriber().load()

        Returns:
            self

        Raises:
            FileNotFoundError: Если модель не найдена (не был запущен setup.py).
            RuntimeError:      При ошибке загрузки модели.
        """
        from faster_whisper import WhisperModel

        # faster-whisper скачивает модели в папку models--Systran--faster-whisper-{size}
        model_path = WHISPER_DIR / f"models--Systran--faster-whisper-{self.model_size}"

        # Внутри папки модель лежит в snapshots/{hash}/
        snapshots = list((model_path / "snapshots").glob("*")) if (model_path / "snapshots").exists() else []
        if snapshots:
            model_path = snapshots[0]  # берём первый (и единственный) снапшот

        if not model_path.exists():
            raise FileNotFoundError(
                f"Модель Whisper '{self.model_size}' не найдена: {model_path}\n"
                "Запустите setup.py для скачивания модели."
            )

        print(f"  [Transcriber] Загружаю модель из {model_path}...")

        self._model = WhisperModel(
            str(model_path),
            device=self.device,
            compute_type=self.compute_type,
        )

        self.is_loaded = True
        print(f"  [Transcriber] Модель загружена.")
        return self

    def transcribe(self, audio_chunk: np.ndarray) -> TranscriptionResult:
        """
        Транскрибирует один аудио-чанк в текст.
        Автоматически определяет язык среди FR/RU/EN.

        Args:
            audio_chunk: Аудио в формате numpy float32, частота 16000 Hz.

        Returns:
            TranscriptionResult с текстом, языком и уверенностью.

        Raises:
            RuntimeError: Если модель не загружена (не вызван load()).
        """
        if not self.is_loaded or self._model is None:
            raise RuntimeError("Модель не загружена. Вызовите .load() перед транскрипцией.")

        # Предобработка аудио
        audio = self._preprocess(audio_chunk)

        # Транскрибируем
        segments, info = self._model.transcribe(
            audio,
            language=None,               # Автодетект языка
            task="transcribe",
            beam_size=5,
            vad_filter=True,             # Фильтр тишины — не транскрибируем паузы
            vad_parameters={
                "min_silence_duration_ms": 500,
                "speech_pad_ms": 200,
                "threshold": 0.7,  # добавь этот параметр — порог обнаружения речи

            },
            condition_on_previous_text=False,  # Каждый чанк независим
            no_speech_threshold=0.6,  # добавь это — отфильтрует тихие чанки
            log_prob_threshold=-1.0,  # и это — отфильтрует неуверенные результаты
        )

        # Собираем текст из сегментов
        text_parts = [seg.text.strip() for seg in segments]
        full_text = " ".join(text_parts).strip()

        # Определяем уверенность (среднее по сегментам если есть)
        confidence = getattr(info, "language_probability", 1.0)

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            confidence=round(confidence, 3),
        )

    # ─── Внутренние методы ────────────────────────────────────────────────────

    def _preprocess(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Нормализует аудио-чанк перед подачей в модель.
        - Конвертирует в float32
        - Нормализует громкость в диапазон [-1, 1]
        - Убирает постоянную составляющую (DC offset)

        Args:
            audio_chunk: Входящий аудио-массив любого типа.

        Returns:
            Нормализованный numpy float32 массив.
        """
        audio = audio_chunk.astype(np.float32)

        # Убираем DC offset (постоянную составляющую)
        audio = audio - np.mean(audio)

        # Нормализуем громкость если сигнал не слишком тихий
        max_val = np.max(np.abs(audio))
        if max_val < 0.01:
            return np.zeros_like(audio)

        return audio

    def _resolve_device(self, device: str) -> str:
        """
        Определяет устройство для запуска модели.
        При device='auto' выбирает 'cuda' если GPU доступен, иначе 'cpu'.

        Args:
            device: 'auto', 'cuda' или 'cpu'.

        Returns:
            'cuda' или 'cpu'.
        """
        if device != "auto":
            return device

        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _resolve_compute_type(self) -> str:
        """
        Выбирает тип вычислений в зависимости от устройства.
        - GPU (CUDA): float16 — быстро и точно
        - CPU:        int8    — оптимизировано для процессора

        Returns:
            'float16' для GPU или 'int8' для CPU.
        """
        return "float16" if self.device == "cuda" else "int8"
