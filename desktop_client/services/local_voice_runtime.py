"""本地语音运行时。

职责：
1. 接收本地 ASR 文本并发送到云端 AstrBot
2. 聚合云端回复文本
3. 在本地触发 TTS 播放（可开关）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

from ..bridge import InputMessage, MessageBridge
from ..config import ClientConfig

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
        self._tts_speaker = None
        self._tts_ready = False
        self._init_tts_engine()

    def _init_tts_engine(self) -> None:
        """初始化 Qt TTS 引擎。失败时仅记录日志，不中断应用。"""
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

    def shutdown(self) -> None:
        """清理运行时资源。"""
        self._accumulators.clear()
        if self._tts_speaker is not None:
            try:
                self._tts_speaker.stop()
            except Exception:
                pass
