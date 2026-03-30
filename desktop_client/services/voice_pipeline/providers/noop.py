from __future__ import annotations

from ..base import BaseASRProvider, BaseTTSProvider


class NoopASRProvider(BaseASRProvider):
    async def transcribe_file(self, audio_path: str, **kwargs) -> str:
        return ""

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int | None = None, **kwargs) -> str:
        return ""


class NoopTTSProvider(BaseTTSProvider):
    async def synthesize_to_file(self, text: str, output_path: str | None = None, **kwargs) -> str | None:
        return None

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        return None
