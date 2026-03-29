"""本地语音模型适配器基础接口。

你可以基于这些接口接入自己的 ASR / TTS 模型。
"""

from __future__ import annotations

import importlib
import inspect
from abc import ABC, abstractmethod
from typing import Awaitable, Callable, Optional, TypeVar, cast


class BaseASRAdapter(ABC):
    """本地 ASR 适配器接口。"""

    @abstractmethod
    async def start(self, on_text: Callable[[str], Awaitable[None]]) -> None:
        """启动 ASR 流并通过回调返回识别文本。"""

    @abstractmethod
    async def stop(self) -> None:
        """停止 ASR 流。"""


class BaseTTSAdapter(ABC):
    """本地 TTS 适配器接口。"""

    @abstractmethod
    def speak(self, text: str) -> None:
        """播放文本语音。"""

    @abstractmethod
    def stop(self) -> None:
        """停止播放。"""


T = TypeVar("T")


def load_adapter(adapter_path: str, expected_type: type[T]) -> Optional[T]:
    """按 `module:ClassName` 格式动态加载适配器。"""
    if not adapter_path or ":" not in adapter_path:
        return None

    module_name, class_name = adapter_path.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_cls = getattr(module, class_name)

    if inspect.isclass(adapter_cls) and issubclass(adapter_cls, expected_type):
        return cast(T, adapter_cls())

    raise TypeError(
        f"适配器类型不匹配: {adapter_path}, 期望 {expected_type.__name__}"
    )
