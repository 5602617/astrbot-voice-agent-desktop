from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..base import BaseASRProvider
from ..models import ASRProviderConfig
from .sherpa_asr import SherpaASRProvider


class RuntimeASRProvider(BaseASRProvider):
    def __init__(self, config: ASRProviderConfig, logger):
        self.config = config
        self.logger = logger
        self._backend = (config.runtime_backend or 'faster_whisper').lower()
        self._model = None
        self._cancelled = False
        self._sherpa_provider: SherpaASRProvider | None = None

    async def warmup(self) -> None:
        if self._backend in ('sherpa_onnx', 'sherpa_asr'):
            self._sherpa_provider = SherpaASRProvider(self.config, self.logger)
            await self._sherpa_provider.warmup()
        elif self._backend == 'faster_whisper':
            await self._warmup_faster_whisper()
        elif self._backend in ('funasr', 'custom'):
            self.logger.warning(f"Runtime ASR backend '{self._backend}' 当前为扩展骨架")
        else:
            raise ValueError(f"未知 ASR runtime backend: {self._backend}")

    async def _warmup_faster_whisper(self) -> None:
        try:
            from faster_whisper import WhisperModel
        except Exception as exc:
            raise RuntimeError("缺少 faster-whisper 依赖，请安装 `faster-whisper`") from exc

        model_name = self.config.model_path or 'base'
        self.logger.info(f"初始化 Runtime ASR (faster_whisper): model={model_name}, device={self.config.device}")
        self._model = await asyncio.to_thread(
            WhisperModel,
            model_name,
            device=self.config.device or 'cpu',
            compute_type='int8' if (self.config.device or 'cpu') == 'cpu' else 'float16',
        )

    async def shutdown(self) -> None:
        self._model = None
        if self._sherpa_provider is not None:
            await self._sherpa_provider.shutdown()
            self._sherpa_provider = None

    async def transcribe_file(self, audio_path: str, **kwargs) -> str:
        if self._backend in ('sherpa_onnx', 'sherpa_asr'):
            if self._sherpa_provider is None:
                await self.warmup()
            if self._sherpa_provider is None:
                return ''
            return await self._sherpa_provider.transcribe_file(audio_path, **kwargs)
        if self._backend == 'faster_whisper':
            return await self._transcribe_faster_whisper(audio_path)
        return ''

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int | None = None, **kwargs) -> str:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        try:
            return await self.transcribe_file(temp_path)
        finally:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except Exception:
                pass

    async def _transcribe_faster_whisper(self, audio_path: str) -> str:
        if self._model is None:
            await self.warmup()
        if self._model is None:
            return ''

        def _run():
            segments, _ = self._model.transcribe(
                audio_path,
                language=self.config.language or None,
                vad_filter=True,
            )
            return ''.join(seg.text for seg in segments).strip()

        text = await asyncio.to_thread(_run)
        self.logger.info(f"Runtime ASR 完成: text_len={len(text)}")
        return text

    def cancel(self) -> None:
        self._cancelled = True
        if self._sherpa_provider is not None:
            self._sherpa_provider.cancel()
