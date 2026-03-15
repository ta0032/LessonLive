"""
LessonLive — core/session.py
Управление сессией транскрипции.
Хранит историю в памяти и одновременно пишет в .txt файл в реальном времени.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─── Пути ─────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent.resolve()
SESSIONS_DIR = BASE_DIR / "sessions"


# ─── Константы ────────────────────────────────────────────────────────────────

# Метка языка для отображения в файле
LANGUAGE_LABELS = {
    "fr": "FR",
    "ru": "RU",
    "en": "EN",
    "unknown": "??",
}


# ─── Структура одной фразы ────────────────────────────────────────────────────

class Phrase:
    """
    Одна транскрибированная фраза в сессии.

    Атрибуты:
        speaker_id:   'speaker_0', 'speaker_1' или 'unknown'.
        speaker_name: Отображаемое имя ('Антуан', 'Мария' и т.д.).
        text:         Распознанный текст.
        language:     Код языка: 'fr', 'ru', 'en'.
        timestamp:    Время фразы как строка '22:31:05'.
        index:        Порядковый номер фразы в сессии (с 0).
    """

    def __init__(
        self,
        speaker_id: str,
        speaker_name: str,
        text: str,
        language: str,
        timestamp: str,
        index: int,
    ):
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.text = text
        self.language = language
        self.timestamp = timestamp
        self.index = index

    def to_dict(self) -> dict:
        """Сериализует фразу в словарь для отправки через WebSocket."""
        return {
            "type": "phrase",
            "index": self.index,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "language": self.language,
            "timestamp": self.timestamp,
        }

    def to_file_line(self) -> str:
        """
        Форматирует фразу для записи в .txt файл.
        Формат: [22:31:05] Антуан (FR): Alors, aujourd'hui...
        """
        lang_label = LANGUAGE_LABELS.get(self.language, self.language.upper())
        return f"[{self.timestamp}] {self.speaker_name} ({lang_label}): {self.text}\n"


# ─── Класс Session ────────────────────────────────────────────────────────────

class Session:
    """
    Управляет одной сессией транскрипции (одним занятием).
    Хранит все фразы в памяти и дублирует их в .txt файл.

    Пример использования:
        session = Session()
        session.start()
        session.add_phrase("speaker_0", "Антуан", "Bonjour!", "fr")
        session.stop()
    """

    def __init__(self, sessions_dir: Optional[Path] = None):
        """
        Инициализирует новую сессию.
        Генерирует имя файла по текущей дате и времени.

        Args:
            sessions_dir: Директория для хранения файлов.
                          По умолчанию: LessonLive/sessions/
        """
        self.sessions_dir = sessions_dir or SESSIONS_DIR
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        # Генерируем имя файла по дате/времени
        now = datetime.now()
        filename = now.strftime("%Y-%m-%d_%H-%M") + "_lesson.txt"
        self.file_path = self.sessions_dir / filename

        # Состояние сессии
        self.is_active = False
        self.started_at: Optional[datetime] = None
        self.history: list[Phrase] = []
        self._file = None
        self._phrase_counter = 0

        print(f"  [Session] Создана сессия: {filename}")

    # ─── Жизненный цикл ───────────────────────────────────────────────────────

    def start(self):
        """
        Открывает файл для записи и начинает сессию.
        Пишет заголовок с датой и временем начала.

        Raises:
            RuntimeError: Если сессия уже активна.
        """
        if self.is_active:
            raise RuntimeError("Сессия уже активна.")

        self.started_at = datetime.now()
        self._file = open(self.file_path, "w", encoding="utf-8")

        # Заголовок файла
        self._file.write("=" * 60 + "\n")
        self._file.write(f"  LessonLive — Транскрипция занятия\n")
        self._file.write(f"  Начало: {self.started_at.strftime('%d.%m.%Y %H:%M:%S')}\n")
        self._file.write("=" * 60 + "\n\n")
        self._file.flush()

        self.is_active = True
        self.history = []
        self._phrase_counter = 0
        print(f"  [Session] Запись начата → {self.file_path.name}")

    def stop(self):
        """
        Завершает сессию.
        Пишет итоговую статистику в файл и закрывает его.
        """
        if not self.is_active:
            return

        self.is_active = False
        ended_at = datetime.now()

        if self._file:
            # Статистика по спикерам
            stats = self._compute_stats()
            duration = (ended_at - self.started_at).seconds if self.started_at else 0
            minutes, seconds = divmod(duration, 60)

            self._file.write("\n" + "=" * 60 + "\n")
            self._file.write(f"  Конец: {ended_at.strftime('%d.%m.%Y %H:%M:%S')}\n")
            self._file.write(f"  Длительность: {minutes}м {seconds}с\n")
            self._file.write(f"  Всего фраз: {self._phrase_counter}\n")

            for speaker_name, count in stats.items():
                self._file.write(f"  {speaker_name}: {count} фраз\n")

            self._file.write("=" * 60 + "\n")
            self._file.flush()
            self._file.close()
            self._file = None

        print(f"  [Session] Запись завершена. Фраз: {self._phrase_counter}")

    # ─── Добавление фраз ──────────────────────────────────────────────────────

    def add_phrase(
        self,
        speaker_id: str,
        speaker_name: str,
        text: str,
        language: str,
    ) -> Phrase:
        """
        Добавляет новую фразу в историю сессии.
        Одновременно дописывает строку в .txt файл.

        Args:
            speaker_id:   ID спикера ('speaker_0' или 'speaker_1').
            speaker_name: Отображаемое имя спикера.
            text:         Распознанный текст фразы.
            language:     Код языка ('fr', 'ru', 'en').

        Returns:
            Объект Phrase который можно сериализовать для WebSocket.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        phrase = Phrase(
            speaker_id=speaker_id,
            speaker_name=speaker_name,
            text=text,
            language=language,
            timestamp=timestamp,
            index=self._phrase_counter,
        )

        # Сохраняем в память
        self.history.append(phrase)
        self._phrase_counter += 1

        # Пишем в файл сразу (не буферизуем — данные не потеряются при сбое)
        if self._file and not self._file.closed:
            self._file.write(phrase.to_file_line())
            self._file.flush()

        return phrase

    # ─── Доступ к истории ─────────────────────────────────────────────────────

    def get_history(self) -> list[dict]:
        """
        Возвращает всю историю сессии как список словарей.
        Используется при подключении нового WebSocket клиента —
        чтобы показать уже накопленный текст.

        Returns:
            Список dict от каждой фразы (результат Phrase.to_dict()).
        """
        return [phrase.to_dict() for phrase in self.history]

    def get_phrase_count(self) -> int:
        """Возвращает количество фраз в текущей сессии."""
        return self._phrase_counter

    # ─── Список сохранённых сессий ────────────────────────────────────────────

    @staticmethod
    def get_sessions_list(sessions_dir: Optional[Path] = None) -> list[dict]:
        """
        Возвращает список всех сохранённых файлов транскрипций.
        Отсортирован по дате — новые файлы первые.

        Args:
            sessions_dir: Директория для поиска. По умолчанию: LessonLive/sessions/

        Returns:
            Список словарей: [{"filename": "...", "date": "...", "size_kb": ...}, ...]
        """
        directory = sessions_dir or SESSIONS_DIR

        if not directory.exists():
            return []

        files = sorted(
            directory.glob("*.txt"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,   # Новые — первые
        )

        result = []
        for f in files:
            size_kb = round(f.stat().st_size / 1024, 1)
            # Парсим дату из имени файла (2024-03-14_22-30_lesson.txt)
            try:
                date_str = f.stem.split("_lesson")[0]  # '2024-03-14_22-30'
                date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M")
                date_formatted = date.strftime("%d.%m.%Y %H:%M")
            except ValueError:
                date_formatted = f.stem

            result.append({
                "filename": f.name,
                "date": date_formatted,
                "size_kb": size_kb,
            })

        return result

    @staticmethod
    def read_session_file(filename: str, sessions_dir: Optional[Path] = None) -> str:
        """
        Читает содержимое файла транскрипции по имени.

        Args:
            filename:     Имя файла (например '2024-03-14_22-30_lesson.txt').
            sessions_dir: Директория. По умолчанию: LessonLive/sessions/

        Returns:
            Содержимое файла как строка.

        Raises:
            FileNotFoundError: Если файл не найден.
        """
        directory = sessions_dir or SESSIONS_DIR
        file_path = directory / filename

        if not file_path.exists():
            raise FileNotFoundError(f"Файл не найден: {filename}")

        # Проверяем что путь не выходит за пределы sessions/ (защита от path traversal)
        if not file_path.resolve().is_relative_to(directory.resolve()):
            raise PermissionError("Доступ запрещён.")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    # ─── Вспомогательные методы ───────────────────────────────────────────────

    def _compute_stats(self) -> dict[str, int]:
        """
        Считает количество фраз по каждому спикеру.
        Используется для итоговой статистики в конце файла.

        Returns:
            Словарь: {имя_спикера: количество_фраз}
        """
        stats: dict[str, int] = {}
        for phrase in self.history:
            name = phrase.speaker_name
            stats[name] = stats.get(name, 0) + 1
        return stats
