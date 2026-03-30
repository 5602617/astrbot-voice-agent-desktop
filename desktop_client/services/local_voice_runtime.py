"""本地语音运行时。

职责：
1. 接收本地 ASR 文本并发送到云端 AstrBot
2. 聚合云端回复文本
3. 在本地触发 TTS 播放（可开关）
"""

from __future__ import annotations

import logging
import asyncio
from dataclasses import dataclass
from typing import Dict, Optional

from ..bridge import InputMessage, MessageBridge
from ..config import ClientConfig
from .voice_adapter_base import BaseASRAdapter, BaseTTSAdapter, load_adapter

logger = logging.getLogger(__name__)


@dataclass
class ReplyAccumulator:
    """按请求维度累积流式回复文本。"""

    session_id: str
    text: str = ""


class LocalVoiceRuntime:
    """本地语音运行时模块（与插件解耦，可独立复用）。"""

    def __init__(self, bridge: MessageBridge, config: ClientConfig):
        self._bridge = bridge
        self._config = config
        self._accumulators: Dict[str, ReplyAccumulator] = {}
        self._asr_task: Optional[asyncio.Task] = None
        self._running = False
        self._asr_adapter: Optional[BaseASRAdapter] = None
        self._tts_adapter: Optional[BaseTTSAdapter] = None
        self._tts_speaker = None
        self._tts_ready = False
        self._load_custom_adapters()
        self._init_tts_engine()

    def _load_custom_adapters(self) -> None:
        """按配置加载本地 ASR/TTS 适配器。"""
        asr_adapter_path = getattr(self._config.voice, "local_asr_adapter", "")
        tts_adapter_path = getattr(self._config.voice, "local_tts_adapter", "")

        try:
            self._asr_adapter = load_adapter(asr_adapter_path, BaseASRAdapter)
            if self._asr_adapter:
                logger.info("LocalVoiceRuntime: 已加载 ASR 适配器 %s", asr_adapter_path)
        except Exception as exc:
            logger.error("LocalVoiceRuntime: 加载 ASR 适配器失败: %s", exc)
            self._asr_adapter = None

        try:
            self._tts_adapter = load_adapter(tts_adapter_path, BaseTTSAdapter)
            if self._tts_adapter:
                logger.info("LocalVoiceRuntime: 已加载 TTS 适配器 %s", tts_adapter_path)
        except Exception as exc:
            logger.error("LocalVoiceRuntime: 加载 TTS 适配器失败: %s", exc)
            self._tts_adapter = None

    def _init_tts_engine(self) -> None:
        """初始化 Qt TTS 引擎。失败时仅记录日志，不中断应用。"""
        if self._tts_adapter is not None:
            self._tts_ready = True
            logger.info("LocalVoiceRuntime: 使用自定义 TTS 适配器")
            return

        try:
            from PySide6.QtTextToSpeech import QTextToSpeech

            self._tts_speaker = QTextToSpeech()
            self._tts_ready = True
            logger.info("LocalVoiceRuntime: QTextToSpeech 初始化成功")
        except Exception as exc:
            self._tts_ready = False
            self._tts_speaker = None
            logger.warning("LocalVoiceRuntime: TTS 不可用，已自动降级为文本模式: %s", exc)

    @property
    def tts_enabled(self) -> bool:
        return bool(getattr(self._config.voice, "enable_tts", True))

    def set_tts_enabled(self, enabled: bool) -> None:
        self._config.voice.enable_tts = enabled

    async def submit_asr_text(self, text: str, session_id: Optional[str] = None) -> bool:
        """将本地 ASR 文本发送到云端 AstrBot。"""
        normalized = (text or "").strip()
        if not normalized:
            return False

        try:
            await self._bridge.send_input(
                InputMessage(
                    msg_type="text",
                    content=normalized,
                    session_id=session_id or self._config.session_id or "",
                    metadata={"source": "local_asr"},
                )
            )
            return True
        except Exception as exc:
            logger.error("LocalVoiceRuntime: 提交 ASR 文本失败: %s", exc)
            return False

    def on_reply_chunk(
        self,
        session_id: str,
        message: str,
        streaming: bool,
        metadata: Optional[dict] = None,
    ) -> None:
        """接收回复文本分片。"""
        if not message:
            return

        request_id = (metadata or {}).get("request_id") or "default"
        key = f"{session_id}:{request_id}"
        accumulator = self._accumulators.setdefault(
            key, ReplyAccumulator(session_id=session_id)
        )

        if streaming:
            accumulator.text += message
        else:
            accumulator.text = message

    def on_reply_end(self, session_id: str, metadata: Optional[dict] = None) -> None:
        """回复结束时触发本地 TTS。"""
        request_id = (metadata or {}).get("request_id") or "default"
        key = f"{session_id}:{request_id}"
        accumulator = self._accumulators.pop(key, None)

        if not accumulator or not accumulator.text.strip():
            return

        if not self.tts_enabled:
            logger.debug("LocalVoiceRuntime: TTS 已禁用，跳过播放")
            return

        self._speak_text(accumulator.text.strip())

    def _speak_text(self, text: str) -> None:
        """执行本地 TTS 播放。"""
        if not text:
            return

        if self._tts_adapter is not None:
            try:
                self._tts_adapter.speak(text)
            except Exception as exc:
                logger.error("LocalVoiceRuntime: 自定义 TTS 适配器播放失败: %s", exc)
            return

        if not self._tts_ready or self._tts_speaker is None:
            logger.warning("LocalVoiceRuntime: 无可用 TTS 引擎，已跳过播放")
            return

        try:
            self._tts_speaker.say(text)
        except Exception as exc:
            logger.error("LocalVoiceRuntime: TTS 播放失败（已降级为仅文本）: %s", exc)

    async def emit_local_asr_event(self, text: str, session_id: Optional[str] = None) -> bool:
        """供外部适配器调用的统一入口（当前阶段直接转发）。"""
        return await self.submit_asr_text(text=text, session_id=session_id)

    async def start(self) -> None:
        """启动本地语音运行时（可选自动启动本地 ASR 适配器）。"""
        if self._running:
            return
        self._running = True
        auto_start_asr = bool(getattr(self._config.voice, "auto_start_local_asr", False))
        if auto_start_asr and self._asr_adapter is not None:
            self._asr_task = asyncio.create_task(self._run_asr_adapter())

    async def _run_asr_adapter(self) -> None:
        """运行本地 ASR 适配器并将识别文本提交到云端。"""
        if self._asr_adapter is None:
            return
        try:
            await self._asr_adapter.start(self._on_asr_text)
        except Exception as exc:
            logger.error("LocalVoiceRuntime: ASR 适配器运行失败（已降级）: %s", exc)

    async def _on_asr_text(self, text: str) -> None:
        await self.submit_asr_text(text=text, session_id=self._config.session_id or "")

    def shutdown(self) -> None:
        """清理运行时资源。"""
        self._running = False
        self._accumulators.clear()
        if self._asr_task is not None:
            self._asr_task.cancel()
            self._asr_task = None

        if self._asr_adapter is not None:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._asr_adapter.stop())
            except RuntimeError:
                pass

        if self._tts_adapter is not None:
            try:
                self._tts_adapter.stop()
            except Exception:
                pass

        if self._tts_speaker is not None:
            try:
                self._tts_speaker.stop()
            except Exception:
                pass
