from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..base import BaseASRProvider
from ..models import ASRProviderConfig


class SherpaASRProvider(BaseASRProvider):
    """SherpaASR runtime provider（优先支持项目自定义 SherpaASR 包装器）。"""

    def __init__(self, config: ASRProviderConfig, logger):
        self.config = config
        self.logger = logger
        self._engine = None
        self._cancelled = False

    async def warmup(self) -> None:
        if self._engine is not None:
            return

        if not self.config.model_path and not (self.config.encoder_path and self.config.decoder_path and self.config.joiner_path):
            raise RuntimeError("SherpaASR 初始化失败：缺少 model_path 或 encoder/decoder/joiner 路径")

        model_path, tokens_path = self._resolve_sherpa_paths()
        self.config.model_path = model_path
        self.config.tokens_path = tokens_path

        # 优先使用项目里常见封装：backend.asr.sherpa_asr.SherpaASR
        try:
            from backend.asr.sherpa_asr import SherpaASR  # type: ignore

            def _init_wrapper():
                return SherpaASR(
                    sample_rate=16000,
                    model_path=model_path or None,
                    tokens_path=tokens_path or None,
                )

            self._engine = await asyncio.to_thread(_init_wrapper)
            self.logger.info("SherpaASR provider 初始化成功（backend.asr.sherpa_asr）")
            return
        except Exception:
            pass

        # 回退到 sherpa_onnx 原生导入检测（仅做依赖检查 + 报错提示）
        try:
            import sherpa_onnx  # noqa: F401
        except Exception as exc:
            raise RuntimeError("未安装 sherpa-onnx 或 backend.asr.sherpa_asr 不可用") from exc

        raise RuntimeError(
            "检测到 sherpa_onnx 依赖，但缺少可用的 SherpaASR 封装。"
            "请提供 backend.asr.sherpa_asr.SherpaASR 或扩展 SherpaASRProvider。"
        )

    def _resolve_sherpa_paths(self) -> tuple[str, str]:
        model_raw = str(self.config.model_path or "").strip()
        tokens_raw = str(self.config.tokens_path or "").strip()
        model_candidates = ("model.int8.onnx", "model.onnx", "encoder.int8.onnx")

        model_path = Path(model_raw) if model_raw else None
        search_dir = None
        if model_path and model_path.exists():
            search_dir = model_path.parent if model_path.is_file() else model_path
            if model_path.is_dir():
                for name in model_candidates:
                    p = model_path / name
                    if p.exists():
                        model_path = p
                        break
        elif model_path:
            raise FileNotFoundError(f"SherpaASR model 路径不存在: {model_raw}")

        if model_path is None and self.config.model_path:
            raise FileNotFoundError(f"SherpaASR model 无法解析: {self.config.model_path}")
        if model_path is None:
            raise RuntimeError("SherpaASR 未提供模型目录/文件")

        tokens_path = Path(tokens_raw) if tokens_raw else None
        if tokens_path and not tokens_path.exists():
            tokens_path = None
        if tokens_path is None:
            lookup_dir = search_dir or model_path.parent
            candidate = lookup_dir / "tokens.txt"
            if candidate.exists():
                tokens_path = candidate
        if tokens_path is None:
            lookup_dir = search_dir or model_path.parent
            raise FileNotFoundError(
                f"SherpaASR 缺少 tokens.txt。已在目录查找: {lookup_dir}"
            )

        self.logger.info("SherpaASR 解析模型: model=%s tokens=%s", model_path, tokens_path)
        return str(model_path), str(tokens_path)

    async def shutdown(self) -> None:
        self._engine = None

    async def transcribe_file(self, audio_path: str, **kwargs) -> str:
        if self._engine is None:
            await self.warmup()
        if self._engine is None:
            return ""

        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_path}")

        def _run() -> str:
            if hasattr(self._engine, 'transcribe_file'):
                return str(self._engine.transcribe_file(str(path))).strip()

            import soundfile as sf
            import numpy as np

            audio, sr = sf.read(str(path), dtype='float32')
            if getattr(audio, 'ndim', 1) > 1:
                audio = audio.mean(axis=1)
            if hasattr(self._engine, 'transcribe_np'):
                return str(self._engine.transcribe_np(audio)).strip()
            raise RuntimeError('SherpaASR 引擎缺少 transcribe_file/transcribe_np 方法')

        text = await asyncio.to_thread(_run)
        self.logger.info(f"SherpaASR 转写完成: text_len={len(text)}")
        return text

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int | None = None, **kwargs) -> str:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        try:
            return await self.transcribe_file(temp_path)
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def cancel(self) -> None:
        self._cancelled = True
