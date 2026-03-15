"""
LessonLive — core/diarize.py
Определение кто из участников говорит в данный момент (speaker diarization).
Использует модель pyannote.audio локально без обращения к интернету.
"""

import io
import json
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─── Пути ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.resolve()
PYANNOTE_DIR = BASE_DIR / "models" / "pyannote"
SPEAKER_MAP_FILE = BASE_DIR / "models" / "speaker_map.json"


# ─── Константы ────────────────────────────────────────────────────────────────

SPEAKER_0 = "speaker_0"
SPEAKER_1 = "speaker_1"
UNKNOWN_SPEAKER = "unknown"

DEFAULT_NAMES = {
    SPEAKER_0: {"ru": "Преподаватель", "en": "Teacher"},
    SPEAKER_1: {"ru": "Ученик", "en": "Student"},
}


# ─── Класс Diarizer ───────────────────────────────────────────────────────────

class Diarizer:
    """
    Определяет кто из двух участников занятия говорит в текущем аудио-чанке.
    Использует модель pyannote/speaker-diarization-3.1.

    Пример использования:
        diarizer = Diarizer(token="hf_xxx").load()
        diarizer.set_speaker_name("speaker_0", "Антуан")
        speaker = diarizer.identify_speaker(audio_chunk)
        print(diarizer.get_speaker_name(speaker))
    """

    def __init__(self, token: Optional[str] = None):
        """
        Инициализирует диаризатор.

        Args:
            token: HuggingFace токен. Нужен только при первом скачивании модели.
                   Если модель уже в models/pyannote/ — токен не требуется.
        """
        self.token = token
        self._pipeline = None
        self.is_loaded = False

        # Словарь маппинга: speaker_id → имя
        # Загружается из файла если существует, иначе используются дефолтные имена
        self.speaker_map: dict[str, str] = {}
        self._load_speaker_map()

        print("  [Diarizer] Инициализирован.")

    # ─── Загрузка модели ──────────────────────────────────────────────────────

    def load(self) -> "Diarizer":
        import torch
        from pyannote.audio import Pipeline

        if not PYANNOTE_DIR.exists():
            raise FileNotFoundError(
                f"Модели pyannote не найдены: {PYANNOTE_DIR}\n"
                "Запустите setup.py для скачивания моделей."
            )

        print("  [Diarizer] Загружаю модели pyannote...")

        # Патч для совместимости pyannote 3.x с PyTorch 2.6+
        _orig = torch.load
        torch.load = lambda *args, **kwargs: _orig(
            *args, **{**kwargs, "weights_only": False}
        )

        try:
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                cache_dir=str(PYANNOTE_DIR),
            )
        finally:
            torch.load = _orig  # восстанавливаем в любом случае

        # Переносим на GPU если доступен
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            self._pipeline = self._pipeline.to(torch.device("cuda"))
            print("  [Diarizer] Модель на GPU.")
        else:
            print("  [Diarizer] Модель на CPU.")

        self.is_loaded = True
        return self

    # ─── Определение спикера ──────────────────────────────────────────────────

    def identify_speaker(self, audio_chunk: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Определяет кто говорит в данном аудио-чанке.
        Запускает пайплайн диаризации и возвращает ID доминирующего спикера.

        Args:
            audio_chunk: Аудио float32, частота sample_rate Hz.
            sample_rate: Частота дискретизации (по умолчанию 16000).

        Returns:
            'speaker_0', 'speaker_1' или 'unknown' если речь не обнаружена.
        """
        if not self.is_loaded or self._pipeline is None:
            return UNKNOWN_SPEAKER

        try:
            import torch

            # pyannote ожидает тензор с частотой дискретизации
            waveform = torch.from_numpy(audio_chunk).unsqueeze(0)  # [1, samples]
            audio_dict = {"waveform": waveform, "sample_rate": sample_rate}

            # Запускаем диаризацию
            diarization = self._pipeline(audio_dict, num_speakers=2)

            # Считаем время для каждого спикера
            speaker_durations: dict[str, float] = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration

            if not speaker_durations:
                return UNKNOWN_SPEAKER

            # Возвращаем спикера с наибольшим временем в чанке
            dominant = max(speaker_durations, key=lambda s: speaker_durations[s])

            # Нормализуем имя спикера к нашему формату
            return self._normalize_speaker_id(dominant)

        except Exception as e:
            print(f"  [Diarizer] Ошибка диаризации: {e}")
            return UNKNOWN_SPEAKER

    # ─── Управление именами спикеров ─────────────────────────────────────────

    def set_speaker_name(self, speaker_id: str, name: str):
        """
        Назначает имя спикеру по его ID.
        Сохраняет маппинг в файл для восстановления при следующем запуске.

        Args:
            speaker_id: 'speaker_0' или 'speaker_1'.
            name:       Имя которое будет отображаться в интерфейсе.

        Пример:
            diarizer.set_speaker_name("speaker_0", "Антуан")
            diarizer.set_speaker_name("speaker_1", "Мария")
        """
        if speaker_id not in (SPEAKER_0, SPEAKER_1):
            print(f"  [Diarizer] Неизвестный speaker_id: {speaker_id}")
            return

        self.speaker_map[speaker_id] = name
        self._save_speaker_map()
        print(f"  [Diarizer] Имя сохранено: {speaker_id} → {name}")

    def get_speaker_name(self, speaker_id: str, lang: str = "ru") -> str:
        """
        Возвращает имя спикера по его ID.
        Если имя не назначено вручную — возвращает дефолтное ('Преподаватель'/'Ученик').

        Args:
            speaker_id: 'speaker_0', 'speaker_1' или 'unknown'.
            lang:       Язык для дефолтных имён: 'ru' или 'en'.

        Returns:
            Имя спикера как строка.
        """
        if speaker_id == UNKNOWN_SPEAKER:
            return {"ru": "Неизвестно", "en": "Unknown"}.get(lang, "Unknown")

        # Сначала ищем в пользовательском маппинге
        if speaker_id in self.speaker_map:
            return self.speaker_map[speaker_id]

        # Иначе дефолтное имя
        return DEFAULT_NAMES.get(speaker_id, {}).get(lang, speaker_id)

    def get_all_speakers(self, lang: str = "ru") -> list[dict]:
        """
        Возвращает список всех спикеров с их именами.
        Используется для отображения в интерфейсе.

        Returns:
            Список словарей: [{"id": "speaker_0", "name": "Антуан"}, ...]
        """
        return [
            {"id": sid, "name": self.get_speaker_name(sid, lang)}
            for sid in (SPEAKER_0, SPEAKER_1)
        ]

    # ─── Сохранение/загрузка маппинга ────────────────────────────────────────

    def _save_speaker_map(self):
        """
        Сохраняет словарь speaker_map в JSON файл models/speaker_map.json.
        Позволяет восстановить имена спикеров при следующем запуске.
        """
        try:
            with open(SPEAKER_MAP_FILE, "w", encoding="utf-8") as f:
                json.dump(self.speaker_map, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  [Diarizer] Не удалось сохранить speaker_map: {e}")

    def _load_speaker_map(self):
        """
        Загружает ранее сохранённый speaker_map из JSON файла.
        Если файл не существует — speaker_map остаётся пустым
        и будут использоваться дефолтные имена.
        """
        if not SPEAKER_MAP_FILE.exists():
            return

        try:
            with open(SPEAKER_MAP_FILE, "r", encoding="utf-8") as f:
                self.speaker_map = json.load(f)
            print(f"  [Diarizer] Загружены имена спикеров: {self.speaker_map}")
        except Exception as e:
            print(f"  [Diarizer] Не удалось загрузить speaker_map: {e}")
            self.speaker_map = {}

    # ─── Вспомогательные методы ───────────────────────────────────────────────

    def _normalize_speaker_id(self, raw_id: str) -> str:
        """
        Нормализует ID спикера от pyannote к нашему формату.
        pyannote возвращает 'SPEAKER_00', 'SPEAKER_01' и т.д.
        Мы конвертируем в 'speaker_0', 'speaker_1'.

        Args:
            raw_id: Сырой ID от pyannote ('SPEAKER_00', 'SPEAKER_01', ...).

        Returns:
            'speaker_0' или 'speaker_1'.
        """
        raw_lower = raw_id.lower()

        # Определяем порядковый номер спикера
        if raw_lower.endswith("0") or raw_lower.endswith("00"):
            return SPEAKER_0
        elif raw_lower.endswith("1") or raw_lower.endswith("01"):
            return SPEAKER_1
        else:
            # Если больше 2 спикеров — маппим на ближайший
            return SPEAKER_0
