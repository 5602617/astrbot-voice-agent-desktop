"""最小录音服务：用于 ASR 触发（开始/停止）。"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


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
        logger.info("ASR录音开始")

    def stop(self) -> str:
        if not self._recording:
            raise AudioRecorderError("录音尚未开始")

        logger.info("ASR录音停止请求")
        assert self._stream is not None
        self._stream.stop_stream()

        # 等待流进入 inactive，避免 Windows 下文件写入/设备句柄释放竞争
        wait_steps = 25  # 25 * 20ms = 500ms
        for _ in range(wait_steps):
            try:
                if not self._stream.is_active():
                    break
            except Exception:
                break
            time.sleep(0.02)

        self._stream.close()
        self._stream = None

        assert self._pa is not None
        self._pa.terminate()
        self._pa = None

        self._recording = False

        fd, path = tempfile.mkstemp(prefix="asr_record_", suffix=".wav")
        os.close(fd)
        Path(path).unlink(missing_ok=True)
        with wave.open(path, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(b"".join(self._frames))

        logger.info("ASR录音文件关闭完成: %s", path)
        return path
