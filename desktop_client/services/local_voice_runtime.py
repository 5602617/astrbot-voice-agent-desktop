"""兼容层：旧 LocalVoiceRuntime 委托到新的 VoicePipelineRuntime。"""

from __future__ import annotations

import asyncio
import contextlib
import logging
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
        self._pending_tasks: set[asyncio.Task] = set()
        self._reply_end_tasks: dict[str, asyncio.Task] = {}

    @property
    def tts_enabled(self) -> bool:
        return bool(getattr(self._config.voice, 'enable_tts', True))

    def set_tts_enabled(self, enabled: bool) -> None:
        self._config.voice.enable_tts = enabled
        self._config.voice.tts_enabled = enabled

    def _spawn_task(self, coro) -> None:
        try:
            task = asyncio.create_task(coro)
        except RuntimeError:
            logger.warning("LocalVoiceRuntime 无运行中的事件循环，忽略异步任务创建")
            return
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def _cancel_pending_tasks(self) -> None:
        tasks = [task for task in self._pending_tasks if not task.done()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._pending_tasks.clear()

    async def _cancel_reply_end_task(self, session_id: str) -> None:
        task = self._reply_end_tasks.get(session_id)
        current = asyncio.current_task()

        if task is None:
            return

        if task is current:
            return

        if not task.done():
            logger.info("LocalVoiceRuntime 取消会话 reply_end 任务: session=%s", session_id)
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        if self._reply_end_tasks.get(session_id) is task:
            self._reply_end_tasks.pop(session_id, None)

    async def _cancel_all_reply_end_tasks(self) -> None:
        tasks = list(self._reply_end_tasks.items())
        self._reply_end_tasks.clear()

        for session_id, task in tasks:
            if task.done():
                continue
            logger.info("LocalVoiceRuntime 关闭时取消 reply_end 任务: session=%s", session_id)
            task.cancel()

        for _, task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task

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
        self._spawn_task(
            self._pipeline.handle_llm_message_event(
                msg_type='text',
                message=message,
                streaming=streaming,
                metadata=metadata or {},
                session_id=session_id,
            )
        )

    def on_reply_end(self, session_id: str, metadata: Optional[dict] = None) -> None:
        async def runner():
            await self._cancel_reply_end_task(session_id)
            await self._pipeline.handle_llm_message_event(
                msg_type='end',
                message='',
                streaming=False,
                metadata=metadata or {},
                session_id=session_id,
            )

        try:
            task = asyncio.create_task(runner(), name=f"voice_reply_end:{session_id}")
        except RuntimeError:
            logger.warning("LocalVoiceRuntime 无运行中的事件循环，忽略 reply_end 异步任务")
            return

        self._pending_tasks.add(task)
        self._reply_end_tasks[session_id] = task

        def _on_done(done_task: asyncio.Task) -> None:
            self._pending_tasks.discard(done_task)
            if self._reply_end_tasks.get(session_id) is done_task:
                self._reply_end_tasks.pop(session_id, None)

        task.add_done_callback(_on_done)
    async def emit_local_asr_event(self, text: str, session_id: Optional[str] = None) -> bool:
        return await self.submit_asr_text(text=text, session_id=session_id)

    def interrupt_current_turn(self, session_id: Optional[str] = None, reason: str = 'manual') -> None:
        if session_id:
            self._spawn_task(self._cancel_reply_end_task(session_id))
        self._pipeline.interrupt_current_turn(session_ctx=session_id, reason=reason)

    def stop_tts(self, session_id: Optional[str] = None) -> None:
        self._pipeline.stop_tts(session_ctx=session_id)

    def set_audio_generated_callback(self, callback) -> None:
        self._pipeline.set_audio_generated_callback(callback)

    def set_tts_segment_callback(self, callback) -> None:
        self._pipeline.set_tts_segment_callback(callback)

    async def get_gpt_sovits_status(self) -> dict:
        try:
            provider = getattr(self._pipeline.tts_provider, "_gpt_sovits_provider", None)

            backend = str(getattr(self._config.voice, "runtime_tts_backend", "") or "").lower()
            if provider is None and backend == "gpt_sovits":
                await self._pipeline.reload_from_config()
                provider = getattr(self._pipeline.tts_provider, "_gpt_sovits_provider", None)

            if provider is None:
                return {
                    "running": False,
                    "process_alive": False,
                    "pid": None,
                    "health_path": None,
                    "base_url": "",
                    "detail": "GPT-SoVITS provider 未初始化",
                }

            if hasattr(provider, "get_status"):
                return await provider.get_status()

            return {
                "running": False,
                "process_alive": False,
                "pid": None,
                "health_path": None,
                "base_url": "",
                "detail": "provider 不支持状态查询",
            }
        except Exception as e:
            logger.exception("查询 GPT-SoVITS 状态失败")
            return {
                "running": False,
                "process_alive": False,
                "pid": None,
                "health_path": None,
                "base_url": "",
                "detail": f"查询异常: {e}",
            }

    async def start_gpt_sovits(self) -> bool:
        try:
            provider = getattr(self._pipeline.tts_provider, "_gpt_sovits_provider", None)
            if provider is None:
                await self._pipeline.reload_from_config()
                provider = getattr(self._pipeline.tts_provider, "_gpt_sovits_provider", None)

            if provider is None:
                return False

            await provider.warmup()
            return True
        except Exception:
            logger.exception("显式启动 GPT-SoVITS 失败")
            return False

    async def stop_gpt_sovits(self, force: bool = False) -> bool:
        try:
            provider = getattr(self._pipeline.tts_provider, "_gpt_sovits_provider", None)
            if provider is None:
                return True

            if force:
                provider.config.gpt_sovits_auto_shutdown_on_exit = True

            await provider.shutdown()
            return True
        except Exception:
            logger.exception("显式关闭 GPT-SoVITS 失败")
            return False

    async def shutdown_async(self) -> None:
        await self._cancel_all_reply_end_tasks()
        await self._cancel_pending_tasks()
        await self._pipeline.shutdown()

    def shutdown(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.shutdown_async())
        except RuntimeError:
            pass
