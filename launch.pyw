"""
LessonLive — launch.pyw
Единая точка входа. Двойной клик — запускает всё.

Архитектура:
  - launch.pyw запускает app.py и управляет его жизненным циклом
  - app.py — только сервер, без браузера и без декоративного вывода
  - launch.pyw показывает окно с логами, открывает браузер, управляет треем
"""

import sys
import os
import subprocess

# ─── Автоустановка зависимостей ───────────────────────────────────────────────
# Делаем это первым делом, до всех остальных импортов

def _ensure_deps():
    needed = {"psutil": "psutil", "pystray": "pystray", "PIL": "pillow"}
    missing = []
    for mod, pkg in needed.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install"] + missing + ["--quiet"],
            check=False,
        )

_ensure_deps()

# ─── Остальные импорты ────────────────────────────────────────────────────────

import json
import threading
import webbrowser
import time
import urllib.request
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path

BASE_DIR        = Path(__file__).parent.resolve()
SETUP_DONE_FILE = BASE_DIR / ".setup_done"
APP_FILE        = BASE_DIR / "app.py"
SETUP_FILE      = BASE_DIR / "setup.py"
ICON_FILE       = BASE_DIR / "icon.png"
LOCK_FILE       = BASE_DIR / ".launch.lock"

# ─── Глобальное состояние ─────────────────────────────────────────────────────

_server_process = None   # subprocess.Popen для app.py
_tray_icon      = None   # pystray.Icon
_log_window     = None   # LogWindow

# ─── Защита от дублирования ───────────────────────────────────────────────────

def _is_already_running() -> bool:
    if not LOCK_FILE.exists():
        return False
    try:
        import psutil
        pid = int(LOCK_FILE.read_text().strip())
        return psutil.pid_exists(pid)
    except Exception:
        return False

def _write_lock():
    LOCK_FILE.write_text(str(os.getpid()))

def _remove_lock():
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except Exception:
        pass

# ─── Проверки ─────────────────────────────────────────────────────────────────

def _is_setup_complete() -> bool:
    if not SETUP_DONE_FILE.exists():
        return False
    try:
        with open(SETUP_DONE_FILE) as f:
            return json.load(f).get("whisper", False)
    except Exception:
        return False

def _is_server_ready() -> bool:
    try:
        urllib.request.urlopen("http://localhost:8000/api/status", timeout=1)
        return True
    except Exception:
        return False

# ─── Управление сервером ──────────────────────────────────────────────────────

def _start_server_process():
    """Запускает app.py и возвращает Popen объект."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.Popen(
        [sys.executable, "-u", str(APP_FILE)],
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

def _stop_server():
    global _server_process
    if _server_process and _server_process.poll() is None:
        _server_process.terminate()
        try:
            _server_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _server_process.kill()
    _server_process = None

# ─── Иконка в трее ───────────────────────────────────────────────────────────

def _make_icon_image(online: bool = True):
    from PIL import Image, ImageDraw
    if ICON_FILE.exists():
        return Image.open(ICON_FILE)
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = (74, 144, 217) if online else (90, 90, 90)
    draw.ellipse([2, 2, size-2, size-2], fill=(*color, 255))
    draw.text((20, 14), "L", fill=(255, 255, 255, 255))
    return img

def _launch_tray():
    """Запускает pystray иконку. Вызывать в отдельном потоке."""
    global _tray_icon
    import pystray

    def on_open(icon, item):
        if _is_server_ready():
            webbrowser.open("http://localhost:8000")
        else:
            # Сервер не запущен — запускаем заново
            _relaunch()

    def on_quit(icon, item):
        icon.stop()
        _tray_icon = None
        _stop_server()
        _remove_lock()
        os._exit(0)

    menu = pystray.Menu(
        pystray.MenuItem("Открыть LessonLive", on_open, default=True),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Выйти", on_quit),
    )

    _tray_icon = pystray.Icon(
        name="LessonLive",
        icon=_make_icon_image(online=False),
        title="LessonLive — запускается...",
        menu=menu,
    )

    # Мониторинг состояния сервера
    def _monitor():
        last = None
        while _tray_icon:
            alive = _is_server_ready()
            if alive != last:
                last = alive
                try:
                    _tray_icon.icon = _make_icon_image(online=alive)
                    _tray_icon.title = f"LessonLive — {'работает' if alive else 'остановлен'}"
                except Exception:
                    pass
            time.sleep(3)

    threading.Thread(target=_monitor, daemon=True).start()
    _tray_icon.run()

def _relaunch():
    """Перезапускает сервер и показывает окно с логами."""
    global _server_process
    _stop_server()
    _server_process = _start_server_process()
    threading.Thread(target=_show_launch_window, daemon=True).start()

# ─── Окно с логами ────────────────────────────────────────────────────────────

class LogWindow:
    def __init__(self, title: str, subtitle: str):
        self.root = tk.Tk()
        self.root.title("LessonLive")
        self.root.geometry("700x440")
        self.root.resizable(True, True)
        self.root.configure(bg="#0f0f1a")
        self.root.protocol("WM_DELETE_WINDOW", self._on_user_close)
        self.closed = False
        self._setup_ui(title, subtitle)

    def _setup_ui(self, title, subtitle):
        hdr = tk.Frame(self.root, bg="#1a1a2e", pady=14)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, font=("Segoe UI", 16, "bold"),
                 fg="#e8eaf0", bg="#1a1a2e").pack()
        tk.Label(hdr, text=subtitle, font=("Segoe UI", 9),
                 fg="#9aa0b0", bg="#1a1a2e").pack()

        sf = tk.Frame(self.root, bg="#0f0f1a", pady=8)
        sf.pack(fill="x", padx=16)
        self.status = tk.Label(sf, text="Инициализация...",
                               font=("Segoe UI", 9), fg="#9aa0b0",
                               bg="#0f0f1a", anchor="w")
        self.status.pack(fill="x")
        self.bar = ttk.Progressbar(sf, mode="indeterminate", length=660)
        self.bar.pack(fill="x", pady=(4, 0))
        self.bar.start(12)

        self.log = scrolledtext.ScrolledText(
            self.root, font=("Cascadia Code", 8),
            bg="#16213e", fg="#e8eaf0", relief="flat",
            state="disabled", wrap="word",
        )
        self.log.pack(fill="both", expand=True, padx=16, pady=(0, 8))
        self.log.tag_config("ok",   foreground="#27ae60")
        self.log.tag_config("warn", foreground="#f39c12")
        self.log.tag_config("err",  foreground="#e74c3c")
        self.log.tag_config("info", foreground="#9aa0b0")
        self.log.tag_config("dim",  foreground="#3a4060")

    def add(self, text: str, tag: str = "info"):
        if self.closed:
            return
        def _do():
            self.log.config(state="normal")
            self.log.insert("end", text + "\n", tag)
            self.log.see("end")
            self.log.config(state="disabled")
        try:
            self.root.after(0, _do)
        except Exception:
            pass

    def set_status(self, text: str):
        if self.closed:
            return
        try:
            self.root.after(0, lambda: self.status.config(text=text))
        except Exception:
            pass

    def finish(self):
        if self.closed:
            return
        try:
            self.root.after(0, lambda: (
                self.bar.stop(),
                self.bar.config(mode="determinate", value=100)
            ))
        except Exception:
            pass

    def close(self):
        """Тихо закрывает окно (без остановки сервера)."""
        if self.closed:
            return
        self.closed = True
        try:
            self.root.after(0, self.root.destroy)
        except Exception:
            pass

    def _on_user_close(self):
        """Пользователь закрыл окно крестиком — сервер продолжает работать."""
        self.closed = True
        self.root.destroy()
        # Сервер НЕ останавливаем — остаётся иконка в трее

    def run(self):
        self.root.mainloop()

# ─── Сценарии ─────────────────────────────────────────────────────────────────

def _classify_line(line: str) -> str:
    if any(x in line for x in ["✓", "загружен", "готово", "успешно"]):
        return "ok"
    if any(x in line for x in ["⚠", "Warning", "warning", "предупреждение"]):
        return "warn"
    if any(x in line for x in ["✗", "Error", "error", "Ошибка", "ошибка", "Traceback"]):
        return "err"
    return "info"

def _show_setup_window():
    """Показывает окно установки. Блокирует поток до закрытия."""
    win = LogWindow("LessonLive — Установка",
                    "Первоначальная настройка и загрузка моделей")
    win.set_status("Запускаю установку...")

    done = threading.Event()
    result = [False]

    def _run():
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            [sys.executable, str(SETUP_FILE)],
            cwd=str(BASE_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=env,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                win.add(line, _classify_line(line))
        proc.wait()
        result[0] = _is_setup_complete()
        win.finish()
        if result[0]:
            win.set_status("✓ Установка завершена — запускаю приложение...")
            win.add("\n✓ Установка завершена!", "ok")
            time.sleep(1)
            win.close()
        else:
            win.set_status("⚠ Установка завершена с ошибками")
            win.add("\n⚠ Проверьте лог выше.", "warn")
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    win.run()  # блокирует
    done.wait()
    return result[0]

def _show_launch_window():
    """
    Показывает окно с логами запуска app.py.
    Закрывает окно когда сервер готов и браузер открылся.
    Запускается в отдельном потоке.
    """
    global _log_window, _server_process

    win = LogWindow("LessonLive", "Запуск сервера и загрузка AI-моделей...")
    _log_window = win
    win.set_status("Запускаю сервер...")

    models_ready = [False]
    browser_opened = [False]

    def _stream():
        proc = _server_process
        if not proc:
            return
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue

            tag = _classify_line(line)
            win.add(line, tag)

            if "Все модели загружены" in line and not models_ready[0]:
                models_ready[0] = True
                win.set_status("✓ Модели загружены — открываю браузер...")
                win.finish()
                # Открываем браузер
                webbrowser.open("http://localhost:8000")
                browser_opened[0] = True
                # Закрываем окно через секунду
                time.sleep(1.5)
                win.close()

    threading.Thread(target=_stream, daemon=True).start()

    # Запускаем окно в основном потоке этого вызова
    # Но _show_launch_window вызывается из потока, поэтому используем after
    win.run()

def _run_launch_in_main_thread():
    """Запускает сценарий запуска в главном потоке (нужно для tkinter)."""
    global _server_process

    _server_process = _start_server_process()

    # Запускаем трей в отдельном потоке
    threading.Thread(target=_launch_tray, daemon=False).start()

    # Показываем окно с логами (блокирует главный поток tkinter)
    _show_launch_window_main()

def _show_launch_window_main():
    """Версия _show_launch_window для главного потока."""
    global _log_window, _server_process

    win = LogWindow("LessonLive", "Запуск сервера и загрузка AI-моделей...")
    _log_window = win
    win.set_status("Запускаю сервер...")

    def _stream():
        proc = _server_process
        if not proc:
            return
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tag = _classify_line(line)
            win.add(line, tag)

            if "Все модели загружены" in line:
                win.set_status("✓ Модели загружены — открываю браузер...")
                win.finish()
                webbrowser.open("http://localhost:8000")
                time.sleep(1.5)
                win.close()

    threading.Thread(target=_stream, daemon=True).start()
    win.run()

# ─── Точка входа ──────────────────────────────────────────────────────────────

def main():
    import atexit
    atexit.register(_remove_lock)
    atexit.register(_stop_server)

    # Если уже запущен — просто открываем браузер
    if _is_already_running():
        if _is_server_ready():
            webbrowser.open("http://localhost:8000")
        return

    _write_lock()

    # Установка если нужна
    if not _is_setup_complete():
        success = _show_setup_window()
        if not success:
            return  # установка не завершена — выходим

    # Запуск приложения
    _run_launch_in_main_thread()


if __name__ == "__main__":
    main()
