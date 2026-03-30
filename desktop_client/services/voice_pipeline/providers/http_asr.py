from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx

from ..base import BaseASRProvider
from ..models import ASRProviderConfig, parse_headers_and_params


def _extract_json_key_path(data: Any, key_path: str) -> str:
    cur = data
    for part in key_path.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return ""
    return "" if cur is None else str(cur)


class HTTPASRProvider(BaseASRProvider):
    def __init__(self, config: ASRProviderConfig, logger):
        self.config = config
        self.logger = logger
        self._cancelled = False

    async def transcribe_file(self, audio_path: str, **kwargs) -> str:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"ASR 音频文件不存在: {audio_path}")

        headers, extra = parse_headers_and_params(self.config)
        timeout = httpx.Timeout(self.config.timeout)
        start = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            with path.open('rb') as f:
                files = {self.config.upload_field: (path.name, f, 'audio/wav')}
                resp = await client.post(self.config.api_url, files=files, data=extra, headers=headers)
                resp.raise_for_status()
                data = resp.json()

        text = _extract_json_key_path(data, self.config.response_text_key).strip()
        elapsed = int((time.time() - start) * 1000)
        self.logger.info(f"HTTP ASR 完成: {elapsed}ms, text_len={len(text)}")
        return text

    async def transcribe_bytes(self, audio_bytes: bytes, sample_rate: int | None = None, **kwargs) -> str:
        headers, extra = parse_headers_and_params(self.config)
        payload = dict(extra)
        if sample_rate:
            payload.setdefault('sample_rate', sample_rate)
        timeout = httpx.Timeout(self.config.timeout)
        start = time.time()

        async with httpx.AsyncClient(timeout=timeout) as client:
            files = {self.config.upload_field: ('audio.raw', audio_bytes, 'application/octet-stream')}
            resp = await client.post(self.config.api_url, files=files, data=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        text = _extract_json_key_path(data, self.config.response_text_key).strip()
        elapsed = int((time.time() - start) * 1000)
        self.logger.info(f"HTTP ASR(bytes) 完成: {elapsed}ms, text_len={len(text)}")
        return text

    def cancel(self) -> None:
        self._cancelled = True
