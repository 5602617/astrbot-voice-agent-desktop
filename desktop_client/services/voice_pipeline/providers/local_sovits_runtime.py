from __future__ import annotations

import asyncio
import io
import wave
from pathlib import Path
from typing import Any, Callable


class LocalSovitsRuntime:
    """本地 GPT-SoVITS 运行时（模型目录驱动，懒加载 + 缓存）。"""

    def __init__(self, model_dir: str, logger, timeout: int = 60, device: str | None = None):
        self.model_dir = Path(model_dir).expanduser()
        self.logger = logger
        self.timeout = timeout
        self.device = device or "auto"

        self._lock = asyncio.Lock()
        self._loaded = False
        self._adapter_name = ""
        self._synthesize_impl: Callable[..., Any] | None = None

        self._gpt_weight: Path | None = None
        self._sovits_weight: Path | None = None
        self._config_file: Path | None = None

    def _scan_model_files(self) -> None:
        if not self.model_dir.exists():
            raise FileNotFoundError(f"SoVITS 模型目录不存在: {self.model_dir}")
        if not self.model_dir.is_dir():
            raise NotADirectoryError(f"SoVITS 模型路径不是目录: {self.model_dir}")

        files = list(self.model_dir.rglob("*"))
        ckpts = [p for p in files if p.is_file() and p.suffix.lower() in {".ckpt", ".pt"}]
        pths = [p for p in files if p.is_file() and p.suffix.lower() == ".pth"]
        cfgs = [p for p in files if p.is_file() and p.suffix.lower() in {".yaml", ".yml", ".json"}]

        def _pick(items: list[Path], hints: tuple[str, ...]) -> Path | None:
            for p in items:
                n = p.name.lower()
                if any(h in n for h in hints):
                    return p
            return items[0] if items else None

        self._gpt_weight = _pick(ckpts, ("gpt", "s1", "text"))
        self._sovits_weight = _pick(pths, ("sovits", "gsv", "s2", "vits"))
        self._config_file = _pick(cfgs, ("config", "yaml", "json"))

        if self._gpt_weight is None:
            raise FileNotFoundError(f"SoVITS 模型目录缺少 GPT 权重(.ckpt/.pt): {self.model_dir}")
        if self._sovits_weight is None:
            raise FileNotFoundError(f"SoVITS 模型目录缺少 SoVITS 权重(.pth): {self.model_dir}")
        if self._config_file is None:
            raise FileNotFoundError(f"SoVITS 模型目录缺少配置文件(.yaml/.yml/.json): {self.model_dir}")

        self.logger.info(
            "SoVITS 本地模型扫描: model_dir=%s, gpt=%s, sovits=%s, config=%s",
            self.model_dir,
            self._gpt_weight,
            self._sovits_weight,
            self._config_file,
        )

    async def warmup(self) -> None:
        async with self._lock:
            if self._loaded:
                return
            self._scan_model_files()
            self._setup_adapter()
            self._loaded = True
            self.logger.info("SoVITS 本地运行时加载完成: adapter=%s", self._adapter_name)

    def _setup_adapter(self) -> None:
        errors: list[str] = []

        # 兼容 GPT-SoVITS 官方仓库常见入口
        try:
            from GPT_SoVITS import inference_webui as webui  # type: ignore

            if not hasattr(webui, "set_gpt_weights") or not hasattr(webui, "set_sovits_weights"):
                raise RuntimeError("inference_webui 缺少 set_gpt_weights/set_sovits_weights")
            if not hasattr(webui, "get_tts_wav"):
                raise RuntimeError("inference_webui 缺少 get_tts_wav")

            webui.set_gpt_weights(str(self._gpt_weight))
            webui.set_sovits_weights(str(self._sovits_weight))

            def _synth(
                text: str,
                language: str,
                ref_audio_path: str,
                prompt_text: str,
                prompt_lang: str,
                speaker: str,
            ):
                generator = webui.get_tts_wav(
                    ref_wav_path=ref_audio_path or str(self.model_dir / "reference" / "ref.wav"),
                    prompt_text=prompt_text or "参考文本",
                    prompt_language=prompt_lang or language or "zh",
                    text=text,
                    text_language=language or "zh",
                    how_to_cut="不切",
                    top_k=5,
                    top_p=1.0,
                    temperature=1.0,
                    ref_free=False,
                    speed=1.0,
                )
                return next(generator)

            self._adapter_name = "GPT_SoVITS.inference_webui"
            self._synthesize_impl = _synth
            return
        except Exception as exc:
            errors.append(f"GPT_SoVITS.inference_webui: {exc}")

        joined = "; ".join(errors)
        raise RuntimeError(
            "无法初始化本地 SoVITS 推理，请安装 GPT-SoVITS Python 依赖并确认仓库在 PYTHONPATH。"
            f" model_dir={self.model_dir}; errors={joined}"
        )

    async def synthesize(
        self,
        text: str,
        *,
        language: str = "zh",
        ref_audio_path: str = "",
        prompt_text: str = "",
        prompt_lang: str = "",
        speaker: str = "",
    ) -> bytes:
        if not text.strip():
            return b""

        await self.warmup()

        async with self._lock:
            if self._synthesize_impl is None:
                raise RuntimeError("SoVITS 本地运行时未完成初始化")

            self.logger.info(
                "SoVITS 本地合成开始: model_dir=%s, ref_audio=%s, prompt_lang=%s, language=%s, speaker=%s",
                self.model_dir,
                ref_audio_path or "(empty)",
                prompt_lang or "(empty)",
                language or "(empty)",
                speaker or "(empty)",
            )

            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        self._synthesize_impl,
                        text,
                        language,
                        ref_audio_path,
                        prompt_text,
                        prompt_lang,
                        speaker,
                    ),
                    timeout=max(1, self.timeout),
                )
            except Exception as exc:
                self.logger.exception("SoVITS 本地合成失败: %s", exc)
                raise

            sample_rate, audio = self._normalize_result(result)
            wav_bytes = self._to_wav_bytes(audio, sample_rate)
            self.logger.info("SoVITS 本地合成完成: bytes=%s, sample_rate=%s", len(wav_bytes), sample_rate)
            return wav_bytes

    def _normalize_result(self, result: Any) -> tuple[int, list[float]]:
        if isinstance(result, tuple) and len(result) == 2:
            sample_rate, audio = result
        else:
            raise RuntimeError(f"SoVITS 返回格式不支持: {type(result)}")

        sr = int(sample_rate) if sample_rate else 32000
        try:
            import numpy as np  # type: ignore

            arr = np.asarray(audio)
            if arr.ndim > 1:
                arr = arr[:, 0]
            arr = arr.astype(np.float32)
            arr = np.clip(arr, -1.0, 1.0)
            return sr, arr.tolist()
        except Exception:
            if isinstance(audio, (list, tuple)):
                values = [max(-1.0, min(1.0, float(x))) for x in audio]
                return sr, values
            raise RuntimeError("SoVITS 音频结果解析失败，且 numpy 不可用")

    def _to_wav_bytes(self, audio: list[float], sample_rate: int) -> bytes:
        pcm = bytearray()
        for x in audio:
            iv = int(max(-1.0, min(1.0, float(x))) * 32767.0)
            pcm.extend(int(iv).to_bytes(2, byteorder="little", signed=True))
        bio = io.BytesIO()
        with wave.open(bio, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(pcm))
        return bio.getvalue()
