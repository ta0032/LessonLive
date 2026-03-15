"""
LessonLive — setup.py
Запускается один раз при первой установке.
Устанавливает зависимости, скачивает модели, проверяет окружение.
"""

import os
import sys
import json
import subprocess
import platform
from pathlib import Path

# ─── Пути ────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
WHISPER_DIR = MODELS_DIR / "whisper"
PYANNOTE_DIR = MODELS_DIR / "pyannote"
SESSIONS_DIR = BASE_DIR / "sessions"
ENV_FILE = BASE_DIR / ".env"
SETUP_DONE_FILE = BASE_DIR / ".setup_done"

# ─── Утилиты вывода ───────────────────────────────────────────────────────────

def log(msg: str):
    print(f"  {msg}")

def log_ok(msg: str):
    print(f"  ✓ {msg}")

def log_warn(msg: str):
    print(f"  ⚠ {msg}")

def log_err(msg: str):
    print(f"  ✗ {msg}")

def section(title: str):
    print(f"\n{'─' * 50}")
    print(f"  {title}")
    print(f"{'─' * 50}")

# ─── 1. Проверка версии Python ────────────────────────────────────────────────

def check_python_version() -> bool:
    """
    Проверяет что запущен Python 3.10 или 3.11.
    Завершает установку с ошибкой если версия не подходит.
    """
    section("Проверка Python")
    major, minor = sys.version_info.major, sys.version_info.minor
    log(f"Найден Python {major}.{minor}")

    if major != 3 or minor < 10:
        log_err(f"Требуется Python 3.10 или 3.11. У вас: {major}.{minor}")
        log_err("Скачайте нужную версию: https://www.python.org/downloads/")
        return False

    if minor > 11:
        log_warn(f"Python {major}.{minor} — не тестировалось, возможны проблемы.")
        log_warn("Рекомендуется Python 3.10 или 3.11.")
    else:
        log_ok(f"Python {major}.{minor} — подходит.")

    return True

# ─── 2. Проверка GPU / CUDA ───────────────────────────────────────────────────

def check_cuda() -> bool:
    """
    Проверяет наличие NVIDIA GPU и поддержки CUDA через torch.
    Возвращает True если GPU доступен — это повлияет на выбор
    версии PyTorch и параметр device при загрузке моделей.
    """
    section("Проверка GPU / CUDA")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            log_ok(f"GPU найден: {gpu_name} ({vram:.1f} GB VRAM)")
            log_ok("Модели будут запускаться на GPU — быстрый режим.")
            return True
        else:
            log_warn("NVIDIA GPU не найден или CUDA не установлена.")
            log_warn("Модели будут запускаться на CPU — медленнее (~5–15 сек задержка).")
            return False
    except ImportError:
        # torch ещё не установлен — проверим позже после установки зависимостей
        log("torch ещё не установлен, проверим после установки зависимостей.")
        return False

# ─── 3. Установка зависимостей ────────────────────────────────────────────────

def install_dependencies(has_cuda: bool) -> bool:
    """
    Устанавливает все pip-зависимости из requirements.txt.
    Если есть CUDA — сначала устанавливает PyTorch с поддержкой GPU.
    Возвращает True при успехе.
    """
    section("Установка зависимостей")

    # Устанавливаем PyTorch отдельно чтобы получить нужную версию (CPU или CUDA)
    if has_cuda:
        log("Устанавливаю PyTorch с поддержкой CUDA...")
        result = subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cu118",
            "--quiet"
        ])
        if result.returncode != 0:
            log_warn("Не удалось установить CUDA-версию PyTorch. Пробую CPU-версию...")
            has_cuda = False

    if not has_cuda:
        log("Устанавливаю PyTorch (CPU)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "torch", "torchaudio",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet"
        ])

    # Устанавливаем остальные зависимости
    req_file = BASE_DIR / "requirements.txt"
    if not req_file.exists():
        log_err("Файл requirements.txt не найден!")
        return False

    log("Устанавливаю остальные зависимости...")
    result = subprocess.run([
        sys.executable, "-m", "pip", "install",
        "-r", str(req_file),
        "--quiet"
    ])

    if result.returncode != 0:
        log_err("Ошибка при установке зависимостей.")
        return False

    log_ok("Все зависимости установлены.")
    return True

# ─── 4. Создание папок ────────────────────────────────────────────────────────

def create_directories():
    """
    Создаёт все необходимые папки если они не существуют.
    """
    section("Создание папок")
    for directory in [MODELS_DIR, WHISPER_DIR, PYANNOTE_DIR, SESSIONS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
        log_ok(f"Папка готова: {directory.relative_to(BASE_DIR)}")

# ─── 5. HuggingFace токен ─────────────────────────────────────────────────────

def get_huggingface_token() -> str | None:
    """
    Запрашивает HuggingFace токен у пользователя интерактивно.
    Токен нужен для однократного скачивания модели pyannote.
    После скачивания токен больше не используется в работе приложения.
    Сохраняет токен в .env файл как HF_TOKEN=xxx.
    Возвращает токен или None если пользователь пропустил шаг.
    """
    section("HuggingFace токен (для pyannote)")
    print()
    print("  Для скачивания модели разделения спикеров нужен бесплатный токен.")
    print("  Регистрация: https://huggingface.co/join")
    print("  Токен: https://huggingface.co/settings/tokens")
    print("  После скачивания модели токен можно удалить из .env")
    print()

    # Проверяем не сохранён ли уже токен
    if ENV_FILE.exists():
        existing = _read_env_token()
        if existing:
            log_ok("Токен уже сохранён в .env")
            reuse = input("  Использовать существующий токен? [Y/n]: ").strip().lower()
            if reuse != "n":
                return existing

    token = input("  Вставьте токен (начинается с hf_): ").strip()

    if not token:
        log_warn("Токен не введён. Модель pyannote не будет скачана.")
        log_warn("Запустите setup.py снова чтобы добавить токен позже.")
        return None

    if not token.startswith("hf_"):
        log_warn("Токен должен начинаться с 'hf_'. Проверьте правильность.")

    # Сохраняем в .env
    _save_env_token(token)
    log_ok("Токен сохранён в .env")
    return token

def _save_env_token(token: str):
    """Записывает HF_TOKEN в .env файл."""
    lines = []
    if ENV_FILE.exists():
        with open(ENV_FILE, "r") as f:
            lines = [l for l in f.readlines() if not l.startswith("HF_TOKEN=")]
    lines.append(f"HF_TOKEN={token}\n")
    with open(ENV_FILE, "w") as f:
        f.writelines(lines)

def _read_env_token() -> str | None:
    """Читает HF_TOKEN из .env файла."""
    if not ENV_FILE.exists():
        return None
    with open(ENV_FILE, "r") as f:
        for line in f:
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None

# ─── 6. Скачивание Whisper ────────────────────────────────────────────────────

def download_whisper_model(model_size: str = "medium") -> bool:
    """
    Скачивает модель faster-whisper в папку models/whisper/.
    model_size: 'small' (~500 MB), 'medium' (~1.5 GB), 'large-v3' (~3 GB)
    Модель скачивается один раз и кэшируется локально.
    Возвращает True при успехе.
    """
    section(f"Скачивание модели Whisper ({model_size})")

    # Проверяем не скачана ли уже
    model_path = WHISPER_DIR / model_size
    if model_path.exists() and any(model_path.iterdir()):
        log_ok(f"Модель '{model_size}' уже скачана: {model_path}")
        return True

    log(f"Скачиваю модель '{model_size}'. Это займёт несколько минут...")

    try:
        # Устанавливаем переменную окружения чтобы модель сохранилась в нашу папку
        os.environ["HF_HOME"] = str(PYANNOTE_DIR)

        from faster_whisper import WhisperModel

        # Инициируем скачивание — модель сохранится в cache_dir
        model = WhisperModel(
            model_size,
            device="cpu",           # При скачивании используем CPU
            compute_type="int8",
            download_root=str(WHISPER_DIR)
        )
        del model  # Выгружаем из памяти, нам нужны только файлы

        log_ok(f"Модель Whisper '{model_size}' скачана.")
        return True

    except Exception as e:
        log_err(f"Ошибка при скачивании Whisper: {e}")
        return False

# ─── 7. Скачивание pyannote ───────────────────────────────────────────────────

def download_pyannote_models(token: str) -> bool:
    """
    Скачивает модели pyannote для разделения спикеров.
    Использует HuggingFace токен для доступа к моделям.
    Модели сохраняются в models/pyannote/.
    Возвращает True при успехе.
    """
    section("Скачивание моделей pyannote (разделение спикеров)")

    if not token:
        log_warn("Токен не предоставлен — пропускаю скачивание pyannote.")
        log_warn("Разделение спикеров будет недоступно.")
        return False

    # Проверяем не скачаны ли уже
    config_path = PYANNOTE_DIR / "config.yaml"
    if config_path.exists():
        log_ok("Модели pyannote уже скачаны.")
        return True

    log("Скачиваю модели pyannote. Это займёт несколько минут...")

    try:
        import torch
        import torch.serialization

        _original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: _original_torch_load(
            *args, **{**kwargs, "weights_only": False}
        )

        os.environ["HF_HOME"] = str(PYANNOTE_DIR)
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token

        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            cache_dir=str(PYANNOTE_DIR)
        )
        del pipeline

        torch.load = _original_torch_load

        log_ok("Модели pyannote скачаны.")
        return True

    except Exception as e:
        log_err(f"Ошибка при скачивании pyannote: {e}")
        log_err("Проверьте токен и примите лицензионное соглашение на:")
        log_err("https://huggingface.co/pyannote/speaker-diarization-3.1")
        return False

# ─── 8. Проверка VB-Cable ─────────────────────────────────────────────────────

def check_loopback() -> bool:
    """
    Проверяет наличие WASAPI loopback устройства для захвата системного звука.
    Loopback нужен чтобы слышать голоса учеников из Zoom.
    Возвращает True если loopback доступен.
    """
    section("Проверка WASAPI Loopback (захват звука из Zoom)")

    try:
        import sounddevice as sd
        devices = sd.query_devices()

        loopback_found = False
        for i, device in enumerate(devices):
            name = device.get("name", "")
            if device.get("max_input_channels", 0) > 0:
                if any(kw in name for kw in ["loopback", "Loopback", "Stereo Mix", "Стерео микшер", "стерео микшер", "что слышит"]):
                    log_ok(f"Loopback найден: [{i}] {name}")
                    loopback_found = True

        if not loopback_found:
            log_warn("WASAPI Loopback не найден!")
            print()
            print("  Loopback нужен чтобы слышать голоса учеников из Zoom.")
            print("  Как включить Stereo Mix:")
            print("  1. Правая кнопка на иконке звука в трее → Параметры звука")
            print("  2. Прокрутите вниз → Дополнительные параметры звука")
            print("  3. Вкладка 'Запись' → правая кнопка в пустом месте")
            print("  4. Включите 'Показать отключённые устройства'")
            print("  5. Найдите 'Stereo Mix' → правая кнопка → Включить")
            print("  6. Запустите setup.py снова")
            print()
            print("  Если Stereo Mix недоступен — установите VB-Cable:")
            print("  https://vb-audio.com/Cable/")
            print()

        return loopback_found

    except ImportError:
        log_warn("sounddevice не установлен — пропускаю проверку loopback.")
        return False

# ─── 9. Выбор модели Whisper ──────────────────────────────────────────────────

def choose_whisper_model(has_cuda: bool) -> str:
    """
    Предлагает пользователю выбрать размер модели Whisper
    в зависимости от наличия GPU и объёма VRAM.
    Возвращает строку с размером модели.
    """
    section("Выбор модели Whisper")

    vram_gb = 0
    if has_cuda:
        try:
            import torch
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        except Exception:
            pass

    print()
    print("  Доступные модели:")
    print("  [1] small   — ~500 MB,  быстрая, качество хорошее")
    print("  [2] medium  — ~1.5 GB,  сбалансированная (рекомендуется)")
    print("  [3] large-v3 — ~3 GB,   лучшее качество, медленнее")
    print()

    if has_cuda:
        if vram_gb >= 8:
            default = "3"
            hint = "large-v3 (ваш GPU позволяет)"
        elif vram_gb >= 4:
            default = "2"
            hint = "medium (оптимально для вашего GPU)"
        else:
            default = "1"
            hint = "small (лимит VRAM)"
    else:
        default = "1"
        hint = "small (CPU режим)"

    print(f"  Рекомендуется: {hint}")
    choice = input(f"  Ваш выбор [1/2/3], Enter = {hint}: ").strip()

    model_map = {"1": "small", "2": "medium", "3": "large-v3", "": default}
    chosen = model_map.get(choice, model_map[default])
    log_ok(f"Выбрана модель: {chosen}")
    return chosen

# ─── 10. Финальная проверка и сохранение статуса ─────────────────────────────

def finalize_setup(whisper_ok: bool, pyannote_ok: bool, loopback_ok: bool):
    """
    Сохраняет результаты установки в .setup_done файл.
    Выводит итоговый отчёт что готово, а что нет.
    """
    section("Итоги установки")

    status = {
        "whisper": whisper_ok,
        "pyannote": pyannote_ok,
        "loopback": loopback_ok,
    }

    with open(SETUP_DONE_FILE, "w") as f:
        json.dump(status, f, indent=2)

    print()
    print(f"  {'✓' if whisper_ok else '✗'} Whisper (транскрипция) — {'готово' if whisper_ok else 'не установлен'}")
    print(f"  {'✓' if pyannote_ok else '⚠'} pyannote (разделение спикеров) — {'готово' if pyannote_ok else 'не установлен (опционально)'}")
    print(f"  {'✓' if loopback_ok else '⚠'} WASAPI Loopback — {'найден' if loopback_ok else 'не найден (нужен для голоса учеников)'}")
    print()

    if whisper_ok:
        log_ok("Установка завершена! Запустите приложение: python app.py")
        if not loopback_ok:
            log_warn("Loopback не найден — включите Stereo Mix в настройках звука Windows.")
    else:
        log_err("Whisper не установлен — транскрипция не будет работать.")

# ─── Точка входа ──────────────────────────────────────────────────────────────

def run_setup():
    """
    Главная функция установки.
    Последовательно запускает все шаги.
    Точка входа при запуске файла.
    """
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║          LessonLive — Установка v1.0             ║")
    print("╚══════════════════════════════════════════════════╝")

    # Шаг 1: Проверка Python
    if not check_python_version():
        sys.exit(1)

    # Шаг 2: Проверка GPU (до установки torch — вернёт False, проверим снова позже)
    has_cuda = check_cuda()

    # Шаг 3: Создаём папки
    create_directories()

    # Шаг 4: Установка зависимостей
    if not install_dependencies(has_cuda):
        sys.exit(1)

    # Шаг 5: Перепроверяем CUDA после установки torch
    has_cuda = check_cuda()

    # Шаг 6: Выбор модели Whisper
    model_size = choose_whisper_model(has_cuda)

    # Шаг 7: Скачиваем Whisper
    whisper_ok = download_whisper_model(model_size)

    # Шаг 8: HuggingFace токен
    token = get_huggingface_token()

    # Шаг 9: Скачиваем pyannote
    pyannote_ok = download_pyannote_models(token) if token else False

    # Шаг 10: Проверяем loopback
    loopback_ok = check_loopback()

    # Шаг 11: Итоги
    finalize_setup(whisper_ok, pyannote_ok, loopback_ok)


if __name__ == "__main__":
    run_setup()
