"""本地模型接入模板。

将此文件中的示例类替换为你自己的 ASR/TTS 模型调用逻辑，
然后在 config.json 的 voice 字段中配置：

- "local_asr_adapter": "desktop_client.voice_models.custom_local_models:MyASRAdapter"
- "local_tts_adapter": "desktop_client.voice_models.custom_local_models:MyTTSAdapter"
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from desktop_client.services.voice_adapter_base import BaseASRAdapter, BaseTTSAdapter


class MyASRAdapter(BaseASRAdapter):
    """示例 ASR 适配器。

    你需要把 `_read_next_text()` 替换成自己的识别输出来源：
    - 麦克风流式识别
    - 本地 websocket
    - 本地文件/管道
    """

    def __init__(self):
        self._running = False

    async def start(self, on_text: Callable[[str], Awaitable[None]]) -> None:
        self._running = True
        while self._running:
            text = await self._read_next_text()
            if text:
                await on_text(text)

    async def stop(self) -> None:
        self._running = False

    async def _read_next_text(self) -> str:
        # TODO: 接入你的 ASR 模型输出
        await asyncio.sleep(0.2)
        return ""


class MyTTSAdapter(BaseTTSAdapter):
    """示例 TTS 适配器。"""

    def speak(self, text: str) -> None:
        # TODO: 接入你的 TTS 推理 + 播放流程
        # 例如：wav = model.infer(text); audio_player.play(wav)
        print(f"[MyTTSAdapter] {text}")

    def stop(self) -> None:
        # TODO: 如有播放队列/播放器，在这里停止
        pass
