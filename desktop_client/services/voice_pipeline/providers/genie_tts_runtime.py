from __future__ import annotations

import asyncio
import builtins
from contextlib import contextmanager
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
        self._character = ""

    def validate_config(self) -> list[str]:
        errs: list[str] = []
        mode = getattr(self.config, "genie_mode", getattr(self.config, "mode", "predefined"))
        language = getattr(self.config, "genie_language", getattr(self.config, "language", "zh"))
        predefined_name = getattr(
            self.config,
            "genie_predefined_voice",
            getattr(self.config, "genie_predefined_character_name", getattr(self.config, "predefined_character_name", "")),
        )
        character_name = getattr(self.config, "genie_character_name", getattr(self.config, "character_name", ""))
        model_dir = getattr(self.config, "genie_model_dir", getattr(self.config, "onnx_model_dir", ""))
        ref_audio = getattr(self.config, "genie_reference_audio_path", getattr(self.config, "reference_audio_path", ""))

        if not language:
            errs.append("language 不能为空")
        if mode == "predefined":
            if not predefined_name:
                errs.append("predefined_character_name 不能为空")
        elif mode == "onnx_local":
            if not character_name:
                errs.append("character_name 不能为空")
            if not model_dir:
                errs.append("onnx_model_dir 不能为空")
            elif not Path(model_dir).exists():
                errs.append(f"onnx_model_dir 不存在: {model_dir}")
        else:
            errs.append(f"未知 Genie 模式: {mode}")

        if ref_audio and not Path(ref_audio).exists():
            errs.append(f"reference_audio_path 不存在: {ref_audio}")
        return errs

    async def warmup(self) -> None:
        if self._initialized:
            return
        errs = self.validate_config()
        if errs:
            raise RuntimeError("; ".join(errs))

        genie_data_dir = self._resolve_genie_data_dir()
        os.environ["GENIE_DATA_DIR"] = str(genie_data_dir)
        self._ensure_genie_data_ready(genie_data_dir)

        try:
            import genie_tts as genie  # type: ignore
        except Exception as exc:
            raise RuntimeError("未安装 genie-tts，请先 pip install genie-tts") from exc

        self._genie = genie
        await asyncio.to_thread(self._initialize_character)
        self._initialized = True

    def _resolve_genie_data_dir(self) -> Path:
        configured = (getattr(self.config, "genie_data_dir", "") or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()

        env_value = (os.environ.get("GENIE_DATA_DIR") or "").strip()
        if env_value:
            return Path(env_value).expanduser().resolve()

        return (Path.cwd() / "GenieData").resolve()

    def _ensure_genie_data_ready(self, genie_data_dir: Path) -> None:
        if genie_data_dir.exists() and genie_data_dir.is_dir():
            return
        raise RuntimeError(
            "Genie-TTS 缺少 GenieData 资源目录，已禁用交互式下载。\n"
            f"缺失目录: {genie_data_dir}\n"
            "请先手动准备 GenieData 后重试。可在设置中填写 GENIE_DATA_DIR，"
            "或将 GenieData 放到当前工作目录。"
        )

    @contextmanager
    def _disable_stdin_prompt(self):
        original_input = builtins.input

        def _no_prompt_input(prompt: str = "") -> str:
            raise RuntimeError(
                "检测到 Genie-TTS 尝试进行命令行交互(input)。"
                "桌面客户端已禁用交互式下载，请先手动准备 GenieData。"
            )

        builtins.input = _no_prompt_input
        try:
            yield
        finally:
            builtins.input = original_input

    def _initialize_character(self) -> None:
        assert self._genie is not None
        with self._disable_stdin_prompt():
            mode = getattr(self.config, "genie_mode", getattr(self.config, "mode", "predefined"))
            if mode == "predefined":
                cname = getattr(
                    self.config,
                    "genie_predefined_voice",
                    getattr(self.config, "genie_predefined_character_name", getattr(self.config, "predefined_character_name", "")),
                )
                self._genie.load_predefined_character(cname)
                self._character = cname
            else:
                character_name = getattr(self.config, "genie_character_name", getattr(self.config, "character_name", ""))
                model_dir = getattr(self.config, "genie_model_dir", getattr(self.config, "onnx_model_dir", ""))
                language = getattr(self.config, "genie_language", getattr(self.config, "language", "zh"))
                self._genie.load_character(
                    character_name=character_name,
                    onnx_model_dir=model_dir,
                    language=language,
                )
                self._character = character_name

            ref_audio = getattr(self.config, "genie_reference_audio_path", getattr(self.config, "reference_audio_path", ""))
            ref_text = getattr(self.config, "genie_reference_audio_text", getattr(self.config, "reference_audio_text", ""))
            if ref_audio and ref_text:
                self._genie.set_reference_audio(
                    character_name=self._character,
                    audio_path=ref_audio,
                    audio_text=ref_text,
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
