"""兼容层：旧 LocalVoiceRuntime 委托到新的 VoicePipelineRuntime。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from ..bridge import MessageBridge
from ..config import ClientConfig
from .voice_pipeline.pipeline import VoicePipelineRuntime

logger = logging.getLogger(__name__)


class LocalVoiceRuntime:
    """向后兼容的本地语音运行时包装器。"""

    def __init__(self, bridge: MessageBridge, config: ClientConfig):
        self._bridge = bridge
        self._config = config
        self._pipeline = VoicePipelineRuntime(bridge=bridge, config=config, logger=logger)

    @property
    def tts_enabled(self) -> bool:
        return bool(getattr(self._config.voice, 'enable_tts', True))

    def set_tts_enabled(self, enabled: bool) -> None:
        self._config.voice.enable_tts = enabled
        self._config.voice.tts_enabled = enabled

    async def start(self) -> None:
        await self._pipeline.reload_from_config()

    async def reload_from_config(self) -> None:
        await self._pipeline.reload_from_config()

    async def submit_asr_text(self, text: str, session_id: Optional[str] = None) -> bool:
        if not text.strip():
            return False
        await self._pipeline.handle_asr_text(text, session_ctx=session_id)
        return True

    async def handle_audio_file(self, audio_path: str, session_id: Optional[str] = None) -> Optional[str]:
        return await self._pipeline.handle_audio_file(audio_path, session_ctx=session_id)

    async def handle_audio_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        return await self._pipeline.handle_audio_bytes(
            audio_bytes,
            sample_rate=sample_rate,
            session_ctx=session_id,
        )

    def on_reply_chunk(
        self,
        session_id: str,
        message: str,
        streaming: bool,
        metadata: Optional[dict] = None,
    ) -> None:
        # 兼容旧同步接口：内部转为异步事件处理
        import asyncio

        asyncio.create_task(
            self._pipeline.handle_llm_message_event(
                msg_type='text',
                message=message,
                streaming=streaming,
                metadata=metadata or {},
                session_id=session_id,
            )
        )

    def on_reply_end(self, session_id: str, metadata: Optional[dict] = None) -> None:
        import asyncio

        asyncio.create_task(
            self._pipeline.handle_llm_message_event(
                msg_type='end',
                message='',
                streaming=False,
                metadata=metadata or {},
                session_id=session_id,
            )
        )

    async def emit_local_asr_event(self, text: str, session_id: Optional[str] = None) -> bool:
        return await self.submit_asr_text(text=text, session_id=session_id)

    def interrupt_current_turn(self, session_id: Optional[str] = None, reason: str = 'manual') -> None:
        self._pipeline.interrupt_current_turn(session_ctx=session_id, reason=reason)

    def stop_tts(self, session_id: Optional[str] = None) -> None:
        self._pipeline.stop_tts(session_ctx=session_id)

    def shutdown(self) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._pipeline.shutdown())
        except RuntimeError:
            pass
