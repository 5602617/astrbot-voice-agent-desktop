from __future__ import annotations

import asyncio
import base64
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

import httpx

from ..base import BaseTTSProvider
from ..models import TTSProviderConfig


def _extract_json_key_path(data: dict, key_path: str):
    cur = data
    for part in key_path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class SovitsTTSProvider(BaseTTSProvider):
    """内部 SoVITS/GPT-SoVITS runtime provider（主路径为 HTTP API）。"""

    def __init__(self, config: TTSProviderConfig, logger, cache_dir: Path):
        self.config = config
        self.logger = logger
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def warmup(self) -> None:
        if self.config.api_url:
            self.logger.info("SoVITS runtime 初始化: mode=http api=%s", self.config.api_url)
            return
        if self.config.model_path:
            self.logger.info("SoVITS runtime 初始化: mode=python_api(预留扩展)")
            return
        raise RuntimeError("SoVITS 初始化失败：未配置 tts_api_url（HTTP）或 tts_model_path（Python API 预留）")

    async def synthesize_to_file(
        self,
        text: str,
        output_path: str | None = None,
        **kwargs,
    ) -> str | None:
        if not text.strip():
            return None

        out = Path(output_path) if output_path else self.cache_dir / (
            f"sovits_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.{self.config.audio_format or 'wav'}"
        )
        out.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.config.api_url:
                audio = await self._request_audio_http(text)
            else:
                audio = await self._request_audio_python_api(text)
        except Exception as exc:
            if self.config.fallback_to_pyttsx3:
                self.logger.warning("SoVITS HTTP/PythonAPI 失败，回退 pyttsx3: %s", exc)
                await self._fallback_pyttsx3(text, out)
                return str(out)
            raise

        if not audio:
            raise RuntimeError("SoVITS 未返回有效音频数据")
        out.write_bytes(audio)
        self.logger.info("SoVITS 生成音频: %s (bytes=%s)", out, len(audio))
        return str(out)

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        path = await self.synthesize_to_file(text)
        if not path:
            return None
        return Path(path).read_bytes()

    async def _request_audio_http(self, text: str) -> bytes:
        headers = self._parse_headers()
        payload = self._build_payload(text)
        timeout = httpx.Timeout(float(max(1, self.config.timeout)))

        self.logger.info(
            "SoVITS HTTP 请求: url=%s method=%s response_mode=%s headers=%s",
            self.config.api_url,
            self.config.method,
            self.config.response_mode,
            self._mask_headers_for_log(headers),
        )

        async with httpx.AsyncClient(timeout=timeout) as client:
            if self.config.method == "GET":
                resp = await client.get(self.config.api_url, params=payload, headers=headers)
            else:
                resp = await client.post(self.config.api_url, json=payload, headers=headers)
            resp.raise_for_status()
            return await self._handle_response_audio(resp, client)

    async def _request_audio_python_api(self, text: str) -> bytes:
        raise NotImplementedError("SoVITS Python API 模式尚未实现，请先使用 tts_api_url")

    def _parse_headers(self) -> dict:
        raw = self.config.headers_json or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.logger.error("tts_headers_json 非法 JSON: %s", exc)
            raise ValueError(f"tts_headers_json 非法 JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("tts_headers_json 必须是 JSON object")
        return {str(k): str(v) for k, v in data.items()}

    def _mask_headers_for_log(self, headers: dict) -> dict:
        masked = {}
        for k, v in headers.items():
            lk = str(k).lower()
            if any(token in lk for token in ("authorization", "token", "api-key", "apikey", "key")):
                sv = str(v)
                masked[k] = f"{sv[:4]}***" if len(sv) > 4 else "***"
            else:
                masked[k] = v
        return masked

    def _build_payload(self, text: str) -> dict:
        payload: dict = {}
        raw_extra = self.config.extra_params_json or "{}"
        if raw_extra.strip():
            try:
                extra = json.loads(raw_extra)
            except json.JSONDecodeError as exc:
                self.logger.error("tts_extra_params_json 非法 JSON: %s", exc)
                raise ValueError(f"tts_extra_params_json 非法 JSON: {exc}") from exc
            if not isinstance(extra, dict):
                raise ValueError("tts_extra_params_json 必须是 JSON object")
            payload.update(extra)

        payload[self.config.text_field or "text"] = text
        if self.config.prompt_text:
            payload.setdefault("prompt_text", self.config.prompt_text)
        if self.config.prompt_lang:
            payload.setdefault("prompt_lang", self.config.prompt_lang)
        if self.config.ref_audio_path:
            payload.setdefault("ref_audio_path", self.config.ref_audio_path)
        if self.config.model_path:
            payload.setdefault("model_path", self.config.model_path)
        if self.config.speaker:
            payload.setdefault("speaker", self.config.speaker)
        if self.config.language:
            payload.setdefault("language", self.config.language)
        return payload

    async def _handle_response_audio(self, resp: httpx.Response, client: httpx.AsyncClient) -> bytes:
        mode = (self.config.response_mode or "audio_stream").lower()
        if mode == "audio_stream":
            return resp.content

        data = resp.json()
        if mode == "json_url":
            return await self._handle_json_url_response(data, client)
        if mode == "json_base64":
            return self._handle_json_base64_response(data)
        if mode in ("json_file", "json_path"):
            return self._handle_json_path_response(data)
        raise ValueError(f"不支持的 SoVITS response_mode: {mode}")

    async def _handle_json_url_response(self, data: dict, client: httpx.AsyncClient) -> bytes:
        key = self.config.response_key or "audio_url"
        raw_url = str(_extract_json_key_path(data, key) or "").strip()
        if not raw_url:
            raise ValueError(f"SoVITS json_url 模式未找到响应字段: {key}")
        target_url = raw_url if raw_url.startswith("http") else urljoin(self.config.api_url, raw_url)
        dl = await client.get(target_url)
        dl.raise_for_status()
        return dl.content

    def _handle_json_base64_response(self, data: dict) -> bytes:
        key = self.config.response_key or "audio_base64"
        b64 = str(_extract_json_key_path(data, key) or "").strip()
        if not b64:
            raise ValueError(f"SoVITS json_base64 模式未找到响应字段: {key}")
        return base64.b64decode(b64)

    def _handle_json_path_response(self, data: dict) -> bytes:
        key = self.config.response_key or "audio_path"
        file_path = str(_extract_json_key_path(data, key) or "").strip()
        if not file_path:
            raise ValueError(f"SoVITS json_path 模式未找到响应字段: {key}")
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"SoVITS 返回文件不存在: {file_path}")
        return p.read_bytes()

    async def _fallback_pyttsx3(self, text: str, out: Path) -> None:
        def _run() -> None:
            import pyttsx3

            engine = pyttsx3.init()
            engine.save_to_file(text, str(out))
            engine.runAndWait()

        await asyncio.to_thread(_run)

