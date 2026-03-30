"""SherpaASR 轻量封装。

优先支持 sherpa-onnx SenseVoice 单模型（model + tokens）方案。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class SherpaASR:
    def __init__(
        self,
        sample_rate: int = 16000,
        model_path: str | None = None,
        tokens_path: str | None = None,
        language: str = "zh",
        device: str = "cpu",
    ):
        self.sample_rate = sample_rate
        self.model_path = model_path or ""
        self.tokens_path = tokens_path or ""
        self.language = language
        self.device = device

        if not self.model_path:
            raise ValueError("SherpaASR: model_path 不能为空")
        if not self.tokens_path:
            raise ValueError("SherpaASR: tokens_path 不能为空")
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"SherpaASR: model 不存在: {self.model_path}")
        if not Path(self.tokens_path).exists():
            raise FileNotFoundError(f"SherpaASR: tokens 不存在: {self.tokens_path}")

        try:
            import sherpa_onnx
        except Exception as exc:
            raise RuntimeError("未安装 sherpa-onnx，请先 pip install sherpa-onnx") from exc

        self._sherpa_onnx = sherpa_onnx
        self._recognizer = self._create_recognizer()

    def _create_recognizer(self):
        # 兼容不同版本 API
        if hasattr(self._sherpa_onnx, "OfflineRecognizer"):
            OfflineRecognizer = self._sherpa_onnx.OfflineRecognizer
            # 新版常见 API
            if hasattr(OfflineRecognizer, "from_sense_voice"):
                return OfflineRecognizer.from_sense_voice(
                    model=self.model_path,
                    tokens=self.tokens_path,
                    use_itn=True,
                    language=self.language,
                )
            # 通用构造
            return OfflineRecognizer(
                model=self.model_path,
                tokens=self.tokens_path,
                use_itn=True,
            )
        raise RuntimeError("当前 sherpa-onnx 版本缺少 OfflineRecognizer")

    def transcribe_file(self, path: str) -> str:
        import soundfile as sf

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"SherpaASR: 音频文件不存在: {path}")

        audio, sr = sf.read(str(p), dtype="float32")
        if getattr(audio, "ndim", 1) > 1:
            audio = np.mean(audio, axis=1)

        if sr != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except Exception as exc:
                raise RuntimeError(
                    f"SherpaASR: 重采样失败 (from {sr} to {self.sample_rate})"
                ) from exc

        return self.transcribe_np(audio)

    def transcribe_np(self, audio: np.ndarray) -> str:
        if audio is None or len(audio) == 0:
            return ""

        # 兼容常见 sherpa-onnx 离线识别调用
        stream = self._recognizer.create_stream()
        stream.accept_waveform(self.sample_rate, audio)
        self._recognizer.decode_stream(stream)
        result = stream.result
        text = getattr(result, "text", "") if result is not None else ""
        return str(text or "").strip()
