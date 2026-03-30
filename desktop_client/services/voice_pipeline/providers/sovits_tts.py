from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

import httpx

from ..base import BaseTTSProvider
from ..models import TTSProviderConfig
from .local_sovits_runtime import LocalSovitsRuntime


def _extract_json_value(data: dict, candidate_keys: tuple[str, ...]) -> str:
    for key in candidate_keys:
        cur = data
        ok = True
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                ok = False
                break
        if ok and cur:
            return str(cur).strip()
    return ""


class SovitsTTSProvider(BaseTTSProvider):
    """SoVITS/GPT-SoVITS 内部 runtime provider（本地优先，HTTP 兼容）。"""

    def __init__(self, config: TTSProviderConfig, logger, cache_dir: Path):
        self.config = config
        self.logger = logger
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._local_runtime: LocalSovitsRuntime | None = None
        self._use_local = False

    async def warmup(self) -> None:
        model_dir = (self.config.model_path or "").strip()
        api_url = (self.config.api_url or "").strip()
        if model_dir:
            self._use_local = True
            self._local_runtime = LocalSovitsRuntime(
                model_dir=model_dir,
                logger=self.logger,
                timeout=int(self.config.timeout or 60),
            )
            await self._local_runtime.warmup()
            self.logger.info("SoVITS provider warmup: mode=local model_dir=%s", model_dir)
            return

        if api_url:
            self._use_local = False
            self.logger.info("SoVITS provider warmup: mode=http api_url=%s", api_url)
            return

        raise RuntimeError("SoVITS 初始化失败：请配置 tts_model_path（本地）或 tts_api_url（HTTP兼容）")

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str | None = None,
        **kwargs,
    ) -> str | None:
        if not text.strip():
            return None

        out = Path(output_path) if output_path else self.cache_dir / f"sovits_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.wav"
        out.parent.mkdir(parents=True, exist_ok=True)
        audio = await self.synthesize_bytes(text, **kwargs)
        if not audio:
            raise RuntimeError("SoVITS 未生成音频数据")
        out.write_bytes(audio)
        self.logger.info("SoVITS 音频已写入: %s", out)
        return str(out)

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        if self._local_runtime is None and not (self.config.api_url or "").strip():
            await self.warmup()

        if (self.config.model_path or "").strip():
            if self._local_runtime is None:
                await self.warmup()
            if self._local_runtime is None:
                raise RuntimeError("SoVITS 本地运行时初始化失败")
            self.logger.info("SoVITS synth: mode=local")
            return await self._local_runtime.synthesize(
                text,
                language=self.config.language or "zh",
                ref_audio_path=self.config.ref_audio_path or "",
                prompt_text=self.config.prompt_text or "",
                prompt_lang=self.config.prompt_lang or "",
                speaker=self.config.speaker or "",
            )

        self.logger.info("SoVITS synth: mode=http")
        return await self._synthesize_http(text)

    async def _synthesize_http(self, text: str) -> bytes:
        timeout = httpx.Timeout(float(max(1, int(self.config.timeout or 60))))
        payload = {
            "text": text,
            "text_lang": self.config.language or "zh",
        }
        if self.config.prompt_text:
            payload["prompt_text"] = self.config.prompt_text
        if self.config.prompt_lang:
            payload["prompt_lang"] = self.config.prompt_lang
        if self.config.ref_audio_path:
            payload["ref_audio_path"] = self.config.ref_audio_path
        if self.config.speaker:
            payload["speaker"] = self.config.speaker

        headers = self._parse_headers_json_compat()
        self.logger.info("SoVITS HTTP 请求: url=%s", self.config.api_url)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self.config.api_url, json=payload, headers=headers)
            resp.raise_for_status()

            content_type = (resp.headers.get("content-type") or "").lower()
            if "json" not in content_type:
                return resp.content

            data = resp.json()
            b64 = _extract_json_value(data, ("audio_base64", "data.audio_base64"))
            if b64:
                return base64.b64decode(b64)

            audio_url = _extract_json_value(data, ("audio_url", "data.audio_url", "url", "data.url"))
            if audio_url:
                url = audio_url if audio_url.startswith("http") else urljoin(self.config.api_url, audio_url)
                dl = await client.get(url)
                dl.raise_for_status()
                return dl.content

            audio_path = _extract_json_value(data, ("audio_path", "data.audio_path", "path", "data.path"))
            if audio_path:
                p = Path(audio_path)
                if not p.exists():
                    raise FileNotFoundError(f"SoVITS HTTP 返回的音频路径不存在: {audio_path}")
                return p.read_bytes()

        raise RuntimeError("SoVITS HTTP 返回 JSON 但未包含可识别音频字段(audio_base64/audio_url/audio_path)")

    def _parse_headers_json_compat(self) -> dict:
        raw = getattr(self.config, "headers_json", "") or ""
        if not raw.strip():
            return {}
        try:
            data = json.loads(raw)
        except Exception as exc:
            self.logger.warning("兼容字段 tts_headers_json 非法，已忽略: %s", exc)
            return {}
        if not isinstance(data, dict):
            self.logger.warning("兼容字段 tts_headers_json 非对象，已忽略")
            return {}
        safe = {}
        for k, v in data.items():
            safe[str(k)] = str(v)
        return safe

