"""
LessonLive — core/audio.py
Захват аудио из двух независимых источников:
  - stream_0: микрофон пользователя (преподаватель)
  - stream_1: WASAPI loopback / Стерео микшер (голос учеников из Zoom)

Не требует VB-Cable. Работает на встроенных средствах Windows через WASAPI.
"""

import threading
import numpy as np
import sounddevice as sd
from typing import Callable, Optional


AudioCallback = Callable[[np.ndarray, str], None]


class AudioDevice:
    def __init__(self, index: int, name: str, channels: int, sample_rate: int, is_loopback: bool = False):
        self.index = index
        self.name = name
        self.channels = channels
        self.sample_rate = sample_rate
        self.is_loopback = is_loopback

    def __repr__(self):
        tag = " [LOOPBACK]" if self.is_loopback else ""
        return f"[{self.index}] {self.name}{tag} ({self.channels}ch, {self.sample_rate}Hz)"


class SingleStream:
    def __init__(self, device, source_label, chunk_duration, sample_rate, callback, extra_settings=None):
        self.device = device
        self.source_label = source_label
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self.callback = callback
        self.extra_settings = extra_settings  # ← добавили

        self.chunk_size = int(sample_rate * chunk_duration)
        self._stream = None
        self._native_rate = None
        self._native_chunk_size = None
        self._buffer = np.array([], dtype=np.float32)
        self._lock = threading.Lock()
        self.is_running = False

    def start(self):
        if self.is_running:
            return

        self._buffer = np.array([], dtype=np.float32)
        self._native_rate = int(sd.query_devices(self.device.index)["default_samplerate"])
        self._native_chunk_size = int(self._native_rate * self.chunk_duration)

        print(f"  [AudioStream] Открываю поток: device={self.device.index}, rate={self._native_rate}, extra_settings={self.extra_settings}")

        self._stream = sd.InputStream(
            device=self.device.index,
            channels=1,
            samplerate=self._native_rate,
            dtype="float32",
            callback=self._on_data,
            blocksize=1024,
            extra_settings=self.extra_settings,
        )
        print(f"  [AudioStream] Поток создан, пробую запустить...")
        try:
            self._stream.start()
            print(f"  [AudioStream] Поток запущен успешно!")
        except Exception as e:
            print(f"  [AudioStream] Ошибка при start(): {e}")
            # Пробуем без WasapiSettings
            print(f"  [AudioStream] Пробую без WasapiSettings...")
            self._stream.close()
            self._stream = sd.InputStream(
                device=self.device.index,
                channels=1,
                samplerate=self._native_rate,
                dtype="float32",
                callback=self._on_data,
                blocksize=1024,
            )
            self._stream.start()
            print(f"  [AudioStream] Запущен без WasapiSettings!")

    def stop(self):
        if not self.is_running:
            return

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        with self._lock:
            self._buffer = np.array([], dtype=np.float32)

        self.is_running = False
        print(f"  [AudioStream] '{self.source_label}' остановлен.")

    def _on_data(self, indata, frames, time, status):
        if status:
            print(f"  [AudioStream] '{self.source_label}' предупреждение: {status}")

        mono = indata[:, 0] if indata.ndim > 1 else indata.flatten()

        with self._lock:
            self._buffer = np.concatenate([self._buffer, mono])

            while len(self._buffer) >= self._native_chunk_size:
                chunk = self._buffer[:self._native_chunk_size].copy()
                self._buffer = self._buffer[self._native_chunk_size:]

                # Ресемплинг с нативной частоты до 16000 Hz для Whisper
                if self._native_rate != self.sample_rate:
                    import scipy.signal
                    ratio = self.sample_rate / self._native_rate
                    target_len = int(len(chunk) * ratio)
                    chunk = scipy.signal.resample(chunk, target_len)

                threading.Thread(
                    target=self.callback,
                    args=(chunk, self.source_label),
                    daemon=True
                ).start()


class DualAudioCapture:
    """
    Захватывает аудио из двух источников одновременно:
      - Микрофон → 'microphone' → speaker_0 (преподаватель)
      - Стерео микшер → 'loopback' → speaker_1+ (ученики из Zoom)
    """

    def __init__(self, chunk_duration=3.0, sample_rate=16000):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate

        self._mic_stream = None
        self._loopback_stream = None
        self._callback = None
        self.is_running = False

        self.mic_device = None
        self.loopback_device = None

    @staticmethod
    def _get_mme_hostapi_index() -> int:
        """Возвращает индекс MME в списке host API."""
        for i, api in enumerate(sd.query_hostapis()):
            if api.get("name", "") == "MME":
                return i
        return 0

    def find_default_microphone(self) -> AudioDevice:
        """
        Ищет микрофон среди WASAPI устройств.
        Если WASAPI не найден — берёт любой доступный микрофон.
        """
        mme_idx = self._get_mme_hostapi_index()
        devices = sd.query_devices()

        # Слова которые исключают виртуальные/системные устройства
        exclude_keywords = ["Переназначение", "Первичный", "Input ()", "Primary"]

        for i, device in enumerate(devices):
            name = device["name"]
            if (device.get("max_input_channels", 0) > 0 and
                    device.get("hostapi") == mme_idx and
                    not self._is_loopback_name(name) and
                    not any(kw in name for kw in exclude_keywords)):
                print(f"  [AudioCapture] Микрофон (MME): [{i}] {name}")
                return AudioDevice(
                    index=i,
                    name=name,
                    channels=1,
                    sample_rate=int(device["default_samplerate"]),
                    is_loopback=False,
                )

        raise RuntimeError("Микрофон не найден в системе.")

    def find_loopback_device(self) -> Optional[AudioDevice]:
        """
        Ищет Стерео микшер среди WASAPI устройств.
        Возвращает None если ничего не найдено.
        """
        mme_idx = self._get_mme_hostapi_index()
        devices = sd.query_devices()

        # Запасной вариант — любой Стерео микшер
        for i, device in enumerate(devices):
            if (device.get("max_input_channels", 0) > 0 and
                    self._is_loopback_name(device["name"])):
                print(f"  [AudioCapture] Loopback (fallback): [{i}] {device['name']}")
                return AudioDevice(
                    index=i,
                    name=device["name"],
                    channels=1,
                    sample_rate=int(device["default_samplerate"]),
                    is_loopback=True,
                )

        return None

    def get_microphones(self) -> list:
        """Возвращает список всех доступных микрофонов."""
        result = []
        for i, device in enumerate(sd.query_devices()):
            if (device.get("max_input_channels", 0) > 0 and
                    not self._is_loopback_name(device["name"])):
                result.append(AudioDevice(
                    index=i,
                    name=device["name"],
                    channels=device["max_input_channels"],
                    sample_rate=int(device["default_samplerate"]),
                    is_loopback=False,
                ))
        return result

    def start(self, callback, mic_device=None, loopback_device=None):
        """Запускает захват из микрофона и loopback одновременно."""
        if self.is_running:
            raise RuntimeError("Захват уже запущен.")

        self._callback = callback
        self.mic_device = mic_device or self.find_default_microphone()
        self.loopback_device = loopback_device or self.find_loopback_device()

        self._mic_stream = SingleStream(
            device=self.mic_device,
            source_label="microphone",
            chunk_duration=self.chunk_duration,
            sample_rate=self.sample_rate,
            callback=self._callback,
        )
        self._mic_stream.start()

        if self.loopback_device:
            self._loopback_stream = SingleStream(
                device=self.loopback_device,
                source_label="loopback",
                chunk_duration=self.chunk_duration,
                sample_rate=self.sample_rate,
                callback=self._callback,
            )
            self._loopback_stream.start()
        else:
            print("  [AudioCapture] ⚠ Loopback не найден — голос учеников не захватывается.")

        self.is_running = True

    def stop(self):
        """Останавливает оба потока."""
        if not self.is_running:
            return

        if self._mic_stream:
            self._mic_stream.stop()
            self._mic_stream = None

        if self._loopback_stream:
            self._loopback_stream.stop()
            self._loopback_stream = None

        self.is_running = False
        print("  [AudioCapture] Оба потока остановлены.")

    def get_status(self) -> dict:
        """Возвращает статус захвата для API."""
        return {
            "is_running": self.is_running,
            "microphone": {
                "name": self.mic_device.name if self.mic_device else None,
                "active": self._mic_stream.is_running if self._mic_stream else False,
            },
            "loopback": {
                "name": self.loopback_device.name if self.loopback_device else None,
                "active": self._loopback_stream.is_running if self._loopback_stream else False,
            } if self.loopback_device else None,
        }

    @staticmethod
    def _is_loopback_name(name: str) -> bool:
        """Определяет loopback устройство по названию."""
        keywords = ["loopback", "Loopback", "LOOPBACK", "Stereo Mix", "Стерео микшер"]
        return any(kw in name for kw in keywords)


# Алиас для обратной совместимости
AudioCapture = DualAudioCapture