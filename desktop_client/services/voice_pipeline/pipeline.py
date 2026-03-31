from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Callable, Awaitable, Any

from ...bridge import InputMessage, MessageBridge
from ...config import ClientConfig
from .models import build_runtime_config, VoiceRuntimeConfig
from .registry import create_asr_provider, create_tts_provider
from .turn_manager import VoiceTurnManager, VoiceTurnState
from .providers.noop import NoopTTSProvider


class VoicePipelineRuntime:
    """ASR -> LLM -> TTS 主运行时。"""

    def __init__(self, bridge: MessageBridge, config: ClientConfig, logger):
        self.bridge = bridge
        self.client_config = config
        self.logger = logger
        self.turns = VoiceTurnManager()
        self.runtime_config: VoiceRuntimeConfig = build_runtime_config(config.voice)

        self.asr_provider = create_asr_provider(self.runtime_config.asr, logger)
        self.tts_provider = create_tts_provider(
            self.runtime_config.tts,
            logger,
            cache_dir=self.runtime_config.pipeline.audio_cache_dir,
        )

        self._playing_audio_path: Optional[str] = None
        self._last_reply_started: set[str] = set()
        self._audio_generated_callback: Optional[Callable[[str, str], Awaitable[None]]] = None
        self._tts_chain_lock = asyncio.Lock()

    def set_audio_generated_callback(self, callback: Callable[[str, str], Awaitable[None]]) -> None:
        self._audio_generated_callback = callback

    async def reload_from_config(self) -> None:
        await self.shutdown()
        self.runtime_config = build_runtime_config(self.client_config.voice)
        self.asr_provider = create_asr_provider(self.runtime_config.asr, self.logger)
        self.tts_provider = create_tts_provider(
            self.runtime_config.tts,
            self.logger,
            cache_dir=self.runtime_config.pipeline.audio_cache_dir,
        )
        await self.asr_provider.warmup()
        try:
            await self.tts_provider.warmup()
        except Exception as exc:
            self.logger.error(f"本地 TTS 初始化失败，已自动降级为禁用状态: {exc}")
            self.runtime_config.tts.enabled = False
            self.tts_provider = NoopTTSProvider()
        self.logger.info(
            f"VoicePipeline 重载完成: asr={self.runtime_config.asr.provider_type}/{self.runtime_config.asr.runtime_backend}, "
            f"tts={self.runtime_config.tts.provider_type}/{self.runtime_config.tts.runtime_backend}, "
            f"asr_cls={self.asr_provider.__class__.__name__}, tts_cls={self.tts_provider.__class__.__name__}"
        )

    async def shutdown(self) -> None:
        try:
            await self.asr_provider.shutdown()
        except Exception:
            pass
        try:
            await self.tts_provider.shutdown()
        except Exception:
            pass

    async def handle_audio_file(self, audio_path: str, session_ctx: object | None = None) -> str | None:
        session_id = self._resolve_session_id(session_ctx)
        turn = self.turns.new_turn(session_id)
        self.turns.set_state(session_id, VoiceTurnState.TRANSCRIBING)
        self.logger.info(f"Voice turn transcribing(file): session={session_id}, turn={turn.turn_id}")

        if self.runtime_config.pipeline.interrupt_tts_on_new_input:
            self.stop_tts(session_ctx)

        text = await self.asr_provider.transcribe_file(audio_path)
        if not text:
            return None

        if self.runtime_config.pipeline.emit_asr_text_message:
            self.logger.info(f"ASR 文本: {text[:100]}")

        await self.handle_asr_text(text, session_ctx=session_ctx)
        return text

    async def handle_audio_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: int | None = None,
        session_ctx: object | None = None,
    ) -> str | None:
        session_id = self._resolve_session_id(session_ctx)
        self.turns.new_turn(session_id)
        self.turns.set_state(session_id, VoiceTurnState.TRANSCRIBING)

        if self.runtime_config.pipeline.interrupt_tts_on_new_input:
            self.stop_tts(session_ctx)

        text = await self.asr_provider.transcribe_bytes(audio_bytes, sample_rate=sample_rate)
        if not text:
            return None
        await self.handle_asr_text(text, session_ctx=session_ctx)
        return text

    async def handle_asr_text(self, text: str, session_ctx: object | None = None) -> None:
        session_id = self._resolve_session_id(session_ctx)
        if not text.strip():
            return

        self.turns.set_state(session_id, VoiceTurnState.WAITING_LLM)
        await self.bridge.send_input(
            InputMessage(
                msg_type='text',
                content=text.strip(),
                session_id=session_id,
                metadata={'source': 'voice_pipeline_asr'},
            )
        )

    async def on_llm_reply_start(self, session_ctx: object | None = None) -> None:
        session_id = self._resolve_session_id(session_ctx)
        self.turns.clear_reply(session_id)
        self.turns.set_state(session_id, VoiceTurnState.WAITING_LLM)

    async def on_llm_reply_chunk(self, text: str, session_ctx: object | None = None) -> None:
        session_id = self._resolve_session_id(session_ctx)
        if text:
            self.turns.append_reply(session_id, text)

    async def on_llm_reply_end(
        self,
        full_text: str | None = None,
        session_ctx: object | None = None,
    ) -> str | None:
        session_id = self._resolve_session_id(session_ctx)
        turn = self.turns.current(session_id)
        merged = (full_text or (turn.reply_buffer if turn else '')).strip()
        if not merged:
            self.turns.end_turn(session_id)
            return None

        self.turns.set_state(session_id, VoiceTurnState.SYNTHESIZING)
        out_path = None
        chunks = self._split_tts_text(merged, max_len=15)
        async with self._tts_chain_lock:
            for idx, chunk in enumerate(chunks, start=1):
                self.logger.info("TTS分段合成: session=%s chunk_id=%s/%s text=%s", session_id, idx, len(chunks), chunk)
                cur_out = await self.tts_provider.synthesize_to_file(chunk)
                if cur_out:
                    out_path = cur_out
                    if self.runtime_config.pipeline.auto_play_tts:
                        self.turns.set_state(session_id, VoiceTurnState.PLAYING)
                        await self._notify_audio_generated(session_id, cur_out)
        self.turns.end_turn(session_id)
        return out_path

    def _split_tts_text(self, text: str, max_len: int = 15) -> list[str]:
        txt = (text or "").strip()
        if not txt:
            return []
        punctuation = set("。！？；!?;，,")
        chunks: list[str] = []
        buf = ""
        for ch in txt:
            buf += ch
            if len(buf) >= max_len:
                last_punc_pos = max((buf.rfind(p) for p in punctuation), default=-1)
                if 0 <= last_punc_pos < len(buf):
                    chunks.append(buf[: last_punc_pos + 1].strip())
                    buf = buf[last_punc_pos + 1 :].strip()
                else:
                    chunks.append(buf[:max_len].strip())
                    buf = buf[max_len:].strip()
            elif ch in punctuation and len(buf) <= max_len:
                chunks.append(buf.strip())
                buf = ""
        if buf.strip():
            while len(buf) > max_len:
                chunks.append(buf[:max_len].strip())
                buf = buf[max_len:].strip()
            if buf:
                chunks.append(buf.strip())
        return [c for c in chunks if c]

    def interrupt_current_turn(self, session_ctx: object | None = None, reason: str = 'manual') -> None:
        session_id = self._resolve_session_id(session_ctx)
        self.turns.interrupt(session_id)
        if self.runtime_config.pipeline.interrupt_asr_on_new_input:
            self.asr_provider.cancel()
        if self.runtime_config.pipeline.interrupt_tts_on_new_input:
            self.tts_provider.stop()
        self.logger.info(f"Voice turn interrupted: session={session_id}, reason={reason}")

    def stop_tts(self, session_ctx: object | None = None) -> None:
        self.tts_provider.stop()

    async def handle_llm_message_event(
        self,
        msg_type: str,
        message: str,
        streaming: bool,
        metadata: dict,
        session_id: str,
    ) -> Optional[str]:
        request_id = (metadata or {}).get('request_id', 'default')
        key = f"{session_id}:{request_id}"

        if msg_type == 'text':
            if key not in self._last_reply_started:
                self._last_reply_started.add(key)
                await self.on_llm_reply_start(session_ctx=session_id)
            await self.on_llm_reply_chunk(message, session_ctx=session_id)
            return None

        if msg_type == 'end':
            if key not in self._last_reply_started:
                self.logger.info(f"忽略重复/无效 end 事件: key={key}")
                return None
            self._last_reply_started.discard(key)
            return await self.on_llm_reply_end(session_ctx=session_id)
        return None

    async def _notify_audio_generated(self, session_id: str, audio_path: str) -> None:
        if self._audio_generated_callback:
            await self._audio_generated_callback(session_id, audio_path)
        else:
            self.logger.info(f"TTS 已生成音频（未配置播放回调）: {audio_path}")

    def _resolve_session_id(self, session_ctx: object | None) -> str:
        if isinstance(session_ctx, str) and session_ctx:
            return session_ctx
        return self.client_config.session_id or 'default_session'
