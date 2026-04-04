from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path
from typing import Optional

from ..base import BaseTTSProvider
from ..models import TTSProviderConfig
from .genie_tts_runtime import GenieTTSRuntime
from .gpt_sovits_runtime import GPTSoVITSRuntimeProvider


class RuntimeTTSProvider(BaseTTSProvider):
    def __init__(self, config: TTSProviderConfig, logger, cache_dir: str):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._backend = (config.runtime_backend or 'qt').lower()
        self._qt_speaker = None
        self._pyttsx3 = None
        self._genie_provider: BaseTTSProvider | None = None
        self._gpt_sovits_provider: BaseTTSProvider | None = None
        self._segment_callback = None

    def set_segment_callback(self, callback) -> None:
        self._segment_callback = callback
        self.logger.info("RuntimeTTSProvider 已设置句级回调: backend=%s callback=%r", self._backend, callback)

        if self._gpt_sovits_provider is not None and hasattr(self._gpt_sovits_provider, "set_segment_callback"):
            self._gpt_sovits_provider.set_segment_callback(callback)
            self.logger.info("RuntimeTTSProvider 已透传句级回调到 GPT-SoVITS provider")

    async def warmup(self) -> None:
        if self._backend == 'qt':
            try:
                from PySide6.QtTextToSpeech import QTextToSpeech
                self._qt_speaker = QTextToSpeech()
                self.logger.info('Runtime TTS backend=qt 初始化成功')
            except Exception as exc:
                raise RuntimeError('Qt TTS 初始化失败') from exc
        elif self._backend == 'pyttsx3':
            try:
                import pyttsx3
                self._pyttsx3 = pyttsx3.init()
                self.logger.info('Runtime TTS backend=pyttsx3 初始化成功')
            except Exception as exc:
                raise RuntimeError('pyttsx3 初始化失败') from exc
        elif self._backend == 'edge_tts':
            self.logger.info('Runtime TTS backend=edge_tts 初始化成功')
        elif self._backend == 'genie_tts':
            self._genie_provider = GenieTTSRuntime(self.config, self.logger, str(self.cache_dir))
            await self._genie_provider.warmup()
        elif self._backend == 'gpt_sovits':
            self._gpt_sovits_provider = GPTSoVITSRuntimeProvider(self.config, self.logger, str(self.cache_dir))
            if self._segment_callback and hasattr(self._gpt_sovits_provider, "set_segment_callback"):
                self._gpt_sovits_provider.set_segment_callback(self._segment_callback)
                self.logger.info("RuntimeTTSProvider 在 warmup 后恢复句级回调到 GPT-SoVITS provider")
            await self._gpt_sovits_provider.warmup()
        elif self._backend in ('custom',):
            self.logger.warning(f"Runtime TTS backend '{self._backend}' 当前为扩展骨架")
        else:
            raise ValueError(f'未知 TTS runtime backend: {self._backend}')

    async def shutdown(self) -> None:
        self.stop()
        if self._genie_provider is not None:
            await self._genie_provider.shutdown()
            self._genie_provider = None
        if self._gpt_sovits_provider is not None:
            await self._gpt_sovits_provider.shutdown()
            self._gpt_sovits_provider = None

    async def synthesize_to_file(
            self,
            text: str,
            output_path: Optional[str] = None,
            **kwargs,
    ) -> Optional[str]:
        if not text.strip():
            return None

        out = Path(
            output_path) if output_path else self.cache_dir / f"tts_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)

        if self._backend == 'edge_tts':
            await self._edge_tts_to_file(text, out)
            return str(out)

        if self._backend == 'pyttsx3':
            await self._pyttsx3_to_file(text, out)
            return str(out)

        if self._backend == 'qt':
            # Qt TTS 不稳定支持导出文件，这里回退为仅实时说话
            self._qt_speak(text)
            return None

        if self._backend == 'genie_tts':
            if self._genie_provider is None:
                await self.warmup()
            if self._genie_provider is None:
                return None
            if self.config.auto_play:
                await self._genie_provider.speak(text)
                return None
            return await self._genie_provider.synthesize_to_file(
                text,
                str(out),
                **kwargs,
            )

        if self._backend == 'gpt_sovits':
            if self._gpt_sovits_provider is None:
                await self.warmup()
            if self._gpt_sovits_provider is None:
                return None
            return await self._gpt_sovits_provider.synthesize_to_file(
                text,
                str(out),
                **kwargs,
            )

        return None

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        if self._backend == 'edge_tts':
            path = await self.synthesize_to_file(text)
            if path:
                return Path(path).read_bytes()
        if self._backend == 'genie_tts':
            path = await self.synthesize_to_file(text)
            if path:
                return Path(path).read_bytes()
        if self._backend == 'gpt_sovits':
            path = await self.synthesize_to_file(text)
            if path:
                return Path(path).read_bytes()
        return None

    async def _edge_tts_to_file(self, text: str, out: Path) -> None:
        try:
            import edge_tts
        except Exception as exc:
            raise RuntimeError('缺少 edge-tts 依赖') from exc

        voice = self.config.speaker or 'zh-CN-XiaoxiaoNeural'
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(str(out))

    async def _pyttsx3_to_file(self, text: str, out: Path) -> None:
        if self._pyttsx3 is None:
            await self.warmup()
        if self._pyttsx3 is None:
            return

        def _run():
            self._pyttsx3.save_to_file(text, str(out))
            self._pyttsx3.runAndWait()

        await asyncio.to_thread(_run)

    def _qt_speak(self, text: str) -> None:
        if self._qt_speaker is None:
            try:
                from PySide6.QtTextToSpeech import QTextToSpeech
                self._qt_speaker = QTextToSpeech()
            except Exception:
                return
        self._qt_speaker.say(text)

    def stop(self) -> None:
        if self._qt_speaker is not None:
            try:
                self._qt_speaker.stop()
            except Exception:
                pass

        if self._pyttsx3 is not None:
            try:
                self._pyttsx3.stop()
            except Exception:
                pass

        if self._genie_provider is not None:
            self._genie_provider.stop()

        if self._gpt_sovits_provider is not None:
            self._gpt_sovits_provider.stop()

    def stop_session(self, session_id: str | None = None) -> None:
        if self._qt_speaker is not None:
            try:
                self._qt_speaker.stop()
            except Exception:
                pass

        if self._pyttsx3 is not None:
            try:
                self._pyttsx3.stop()
            except Exception:
                pass

        if self._genie_provider is not None:
            stop_session = getattr(self._genie_provider, "stop_session", None)
            if callable(stop_session):
                stop_session(session_id)
            else:
                self._genie_provider.stop()

        if self._gpt_sovits_provider is not None:
            stop_session = getattr(self._gpt_sovits_provider, "stop_session", None)
            if callable(stop_session):
                stop_session(session_id)
            else:
                self._gpt_sovits_provider.stop()