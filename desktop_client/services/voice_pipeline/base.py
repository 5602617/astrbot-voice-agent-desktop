from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BaseASRProvider(ABC):
    """ASR Provider 抽象接口。"""

    async def warmup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    @abstractmethod
    async def transcribe_file(self, audio_path: str, **kwargs) -> str:
        raise NotImplementedError

    async def transcribe_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: Optional[int] = None,
        **kwargs,
    ) -> str:
        raise NotImplementedError("当前 ASR provider 不支持 transcribe_bytes")

    def cancel(self) -> None:
        return None


class BaseTTSProvider(ABC):
    """TTS Provider 抽象接口。"""

    async def warmup(self) -> None:
        return None

    async def shutdown(self) -> None:
        return None

    @abstractmethod
    async def synthesize_to_file(
        self,
        text: str,
        output_path: Optional[str] = None,
        **kwargs,
    ) -> Optional[str]:
        raise NotImplementedError

    async def synthesize_bytes(self, text: str, **kwargs) -> Optional[bytes]:
        raise NotImplementedError("当前 TTS provider 不支持 synthesize_bytes")

    def stop(self) -> None:
        return None
