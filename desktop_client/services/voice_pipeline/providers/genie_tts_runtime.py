from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from ..base import BaseTTSProvider
from ..models import TTSProviderConfig


class GenieTTSRuntime(BaseTTSProvider):
    """Genie-TTS 本地 runtime 封装（Python API 直连）。"""

    def __init__(self, config: TTSProviderConfig, logger, cache_dir: str):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._genie = None
        self._initialized = False

    def validate_config(self) -> list[str]:
        errs: list[str] = []
        if not self.config.language:
            errs.append("language 不能为空")
        if self.config.genie_mode == "predefined":
            if not self.config.predefined_character_name:
                errs.append("predefined_character_name 不能为空")
        elif self.config.genie_mode == "onnx_local":
            if not self.config.character_name:
                errs.append("character_name 不能为空")
            if not self.config.onnx_model_dir:
                errs.append("onnx_model_dir 不能为空")
            elif not Path(self.config.onnx_model_dir).exists():
                errs.append(f"onnx_model_dir 不存在: {self.config.onnx_model_dir}")
        else:
            errs.append(f"未知 Genie 模式: {self.config.genie_mode}")

        if self.config.reference_audio_path and not Path(self.config.reference_audio_path).exists():
            errs.append(f"reference_audio_path 不存在: {self.config.reference_audio_path}")
        return errs

    async def warmup(self) -> None:
        if self._initialized:
            return
        errs = self.validate_config()
        if errs:
            raise RuntimeError("; ".join(errs))

        if self.config.use_genie_data_dir and self.config.genie_data_dir:
            os.environ["GENIE_DATA_DIR"] = self.config.genie_data_dir

        try:
            import genie_tts as genie  # type: ignore
        except Exception as exc:
            raise RuntimeError("未安装 genie-tts，请先 pip install genie-tts") from exc

        self._genie = genie
        await asyncio.to_thread(self._initialize_character)
        self._initialized = True

    def _initialize_character(self) -> None:
        assert self._genie is not None
        if self.config.genie_mode == "predefined":
            cname = self.config.predefined_character_name
            self._genie.load_predefined_character(cname)
            self._character = cname
        else:
            self._genie.load_character(
                character_name=self.config.character_name,
                onnx_model_dir=self.config.onnx_model_dir,
                language=self.config.language,
            )
            self._character = self.config.character_name

        if self.config.reference_audio_path and self.config.reference_audio_text:
            self._genie.set_reference_audio(
                character_name=self._character,
                audio_path=self.config.reference_audio_path,
                audio_text=self.config.reference_audio_text,
            )

    async def synthesize_to_file(self, text: str, output_path: str | None = None, **kwargs) -> str | None:
        if not text.strip():
            return None
        if not self._initialized:
            await self.warmup()

        if output_path:
            out = Path(output_path)
        else:
            out = self.cache_dir / f"genie_{int(asyncio.get_event_loop().time()*1000)}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)

        await asyncio.to_thread(self._genie.tts, character_name=self._character, text=text, play=False, save_path=str(out))
        return str(out)

    async def speak(self, text: str) -> None:
        if not text.strip():
            return
        if not self._initialized:
            await self.warmup()
        await asyncio.to_thread(self._genie.tts, character_name=self._character, text=text, play=True)
        if hasattr(self._genie, "wait_for_playback_done"):
            await asyncio.to_thread(self._genie.wait_for_playback_done)

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        path = await self.synthesize_to_file(text)
        if not path:
            return None
        return Path(path).read_bytes()

    def stop(self) -> None:
        pass

    async def shutdown(self) -> None:
        self._initialized = False
        self._genie = None
