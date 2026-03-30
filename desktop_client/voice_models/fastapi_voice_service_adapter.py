"""对接本地 FastAPI 语音服务（你贴出的 /asr/wav + /ws 协议）。"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Awaitable, Callable, Optional

import httpx
import websockets

from desktop_client.services.voice_adapter_base import BaseASRAdapter, BaseTTSAdapter


class FastAPIWsASRAdapter(BaseASRAdapter):
    """使用 websocket 与本地 ASR 服务通信。

    默认协议对齐你给出的服务端：
    - ws 地址: ws://127.0.0.1:8000/ws
    - 发送 float32 PCM chunk（二进制）
    - 发送文本控制包: {"type": "mic-audio-end"}

    当服务端回传包含 text 字段时，转发给 on_text。
    """

    def __init__(self, ws_url: str = "ws://127.0.0.1:8000/ws"):
        self.ws_url = ws_url
        self._ws = None
        self._running = False
        self._on_text: Optional[Callable[[str], Awaitable[None]]] = None
        self._recv_task: Optional[asyncio.Task] = None

    async def start(self, on_text: Callable[[str], Awaitable[None]]) -> None:
        self._on_text = on_text
        self._running = True
        self._ws = await websockets.connect(self.ws_url)
        self._recv_task = asyncio.create_task(self._recv_loop())

    async def stop(self) -> None:
        self._running = False
        if self._recv_task:
            self._recv_task.cancel()
            self._recv_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def send_pcm_chunk(self, chunk: bytes) -> None:
        """发送一段 float32 PCM 音频数据。"""
        if self._ws is not None:
            await self._ws.send(chunk)

    async def end_utterance(self) -> None:
        """触发服务端识别一次完整语句。"""
        if self._ws is not None:
            await self._ws.send(json.dumps({"type": "mic-audio-end"}))

    async def reset(self) -> None:
        if self._ws is not None:
            await self._ws.send(json.dumps({"type": "reset"}))

    async def _recv_loop(self) -> None:
        while self._running and self._ws is not None:
            try:
                raw = await self._ws.recv()
            except Exception:
                break

            if not isinstance(raw, str):
                continue

            try:
                data = json.loads(raw)
            except Exception:
                continue

            # 兼容不同返回格式：{"text": "..."} / {"type": "asr_result", "text": "..."}
            text = (data.get("text") or "").strip()
            if text and self._on_text:
                await self._on_text(text)


class FastAPIHttpASRAdapter(BaseASRAdapter):
    """使用 /asr/wav 的 HTTP 方案（适合文件/分段 wav 输入）。"""

    def __init__(self, asr_url: str = "http://127.0.0.1:8000/asr/wav"):
        self.asr_url = asr_url
        self._running = False
        self._on_text: Optional[Callable[[str], Awaitable[None]]] = None

    async def start(self, on_text: Callable[[str], Awaitable[None]]) -> None:
        # 该适配器不主动采集音频，仅保存回调，等待 submit_wav_* 被调用
        self._on_text = on_text
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def submit_wav_file(self, wav_path: str | Path) -> Optional[str]:
        if not self._running:
            return None
        path = Path(wav_path)
        if not path.exists():
            return None

        async with httpx.AsyncClient(timeout=30) as client:
            with path.open("rb") as f:
                files = {"file": (path.name, f, "audio/wav")}
                resp = await client.post(self.asr_url, files=files)
                resp.raise_for_status()
                data = resp.json()

        text = (data.get("text") or "").strip()
        if text and self._on_text:
            await self._on_text(text)
        return text


class FastAPITTSAdapter(BaseTTSAdapter):
    """调用本地 FastAPI TTS 接口。

    默认请求 `POST /tts`，支持你后续按服务端实际结构调整。
    """

    def __init__(self, tts_url: str = "http://127.0.0.1:8000/tts"):
        self.tts_url = tts_url

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        asyncio.create_task(self._speak_async(text))

    async def _speak_async(self, text: str) -> None:
        async with httpx.AsyncClient(timeout=30) as client:
            await client.post(self.tts_url, json={"text": text})

    def stop(self) -> None:
        # 远程 HTTP TTS 无本地播放队列控制，这里保持空实现
        pass
