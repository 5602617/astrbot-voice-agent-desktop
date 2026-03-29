"""本地语音桥接插件。

将本地 ASR 文本接入云端 AstrBot，并在本地执行 TTS。
"""

from __future__ import annotations

import logging
from typing import Optional

from desktop_client.plugins.base import IPlugin, PluginMetadata
from desktop_client.plugins.hooks import HookContext, HookResult, HookType
from desktop_client.services.local_voice_runtime import LocalVoiceRuntime

logger = logging.getLogger(__name__)


class LocalVoiceBridgePlugin(IPlugin):
    """ASR -> 云端 AstrBot -> 本地 TTS 的桥接插件。"""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="local_voice_bridge",
            version="0.1.0",
            author="AstrBot Desktop",
            description="本地 ASR 文本转发到云端 AstrBot，并在本地执行 TTS 播放",
            tags=["voice", "asr", "tts", "cloud-adapter"],
        )

    def on_load(self) -> bool:
        bridge = getattr(self._manager, "app_bridge", None)
        config = getattr(self._manager, "app_config", None)
        if bridge is None or config is None:
            logger.error("LocalVoiceBridgePlugin: 缺少 bridge/config 上下文")
            return False

        self._runtime = LocalVoiceRuntime(bridge=bridge, config=config)

        self.register_hook(HookType.POST_MESSAGE_RECEIVE, self._on_post_message_receive)
        self.load_config()

        if "tts_enabled" in self.config:
            self._runtime.set_tts_enabled(bool(self.config.get("tts_enabled", True)))

        return True

    def on_unload(self) -> None:
        self.set_config_value("tts_enabled", self._runtime.tts_enabled)
        self.save_config()
        super().on_unload()

    def on_enable(self) -> bool:
        import asyncio

        asyncio.create_task(self._runtime.start())
        return True

    def on_disable(self) -> None:
        self._runtime.shutdown()

    async def _on_post_message_receive(self, context: HookContext) -> HookResult:
        msg_type = context.get("msg_type", "")
        session_id = context.get("session_id", "")
        metadata = context.get("metadata", {})

        if msg_type == "text":
            self._runtime.on_reply_chunk(
                session_id=session_id,
                message=context.get("message", ""),
                streaming=bool(context.get("streaming", False)),
                metadata=metadata,
            )
        elif msg_type == "end":
            self._runtime.on_reply_end(session_id=session_id, metadata=metadata)

        return HookResult.CONTINUE

    async def submit_asr_text(self, text: str, session_id: Optional[str] = None) -> bool:
        """供本地 ASR 适配器调用：提交识别文本到云端。"""
        return await self._runtime.submit_asr_text(text=text, session_id=session_id)

    def set_tts_enabled(self, enabled: bool) -> None:
        self._runtime.set_tts_enabled(enabled)
        self.set_config_value("tts_enabled", enabled)
        self.save_config()
