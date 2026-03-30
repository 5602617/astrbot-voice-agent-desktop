"""最小录音服务：用于 ASR 触发（开始/停止）。"""

from __future__ import annotations

import tempfile
import wave
from pathlib import Path
from typing import List


class AudioRecorderError(RuntimeError):
    pass


class AudioRecorderService:
    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._recording = False
        self._frames: List[bytes] = []
        self._stream = None
        self._pa = None

    @property
    def is_recording(self) -> bool:
        return self._recording

    def start(self) -> None:
        if self._recording:
            return

        try:
            import pyaudio
        except Exception as exc:
            raise AudioRecorderError("缺少 pyaudio 依赖，无法录音") from exc

        self._pa = pyaudio.PyAudio()
        self._frames = []

        def _callback(in_data, frame_count, time_info, status):
            self._frames.append(in_data)
            return (None, pyaudio.paContinue)

        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=1024,
            stream_callback=_callback,
        )
        self._stream.start_stream()
        self._recording = True

    def stop(self) -> str:
        if not self._recording:
            raise AudioRecorderError("录音尚未开始")

        assert self._stream is not None
        self._stream.stop_stream()
        self._stream.close()
        self._stream = None

        assert self._pa is not None
        self._pa.terminate()
        self._pa = None

        self._recording = False

        fd, path = tempfile.mkstemp(prefix="asr_record_", suffix=".wav")
        Path(path).unlink(missing_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self._frames))

        return path
