from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from urllib.parse import urljoin

import httpx

from ..base import BaseTTSProvider
from ..models import TTSProviderConfig, parse_headers_and_params


def _extract_json_key_path(data: dict, key_path: str):
    cur = data
    for part in key_path.split('.'):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


class HTTPTTSProvider(BaseTTSProvider):
    def __init__(self, config: TTSProviderConfig, logger, cache_dir: str):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    async def synthesize_to_file(self, text: str, output_path: str | None = None, **kwargs) -> str | None:
        audio = await self.synthesize_bytes(text, **kwargs)
        if not audio:
            return None

        path = Path(output_path) if output_path else self.cache_dir / f"tts_{int(time.time()*1000)}_{uuid.uuid4().hex[:8]}.{self.config.audio_format}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(audio)
        self.logger.info(f"HTTP TTS 生成音频: {path}")
        return str(path)

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        headers, extra = parse_headers_and_params(self.config)
        payload = dict(extra)
        payload[self.config.text_field] = text

        timeout = httpx.Timeout(60)
        start = time.time()
        async with httpx.AsyncClient(timeout=timeout) as client:
            if self.config.method == 'GET':
                resp = await client.get(self.config.api_url, params=payload, headers=headers)
            else:
                resp = await client.post(self.config.api_url, json=payload, headers=headers)
            resp.raise_for_status()

            mode = self.config.response_mode
            if mode == 'audio_stream':
                audio_bytes = resp.content
            else:
                data = resp.json()
                if mode == 'json_url':
                    url = str(_extract_json_key_path(data, self.config.response_key) or '').strip()
                    if not url:
                        return None
                    target_url = url if url.startswith('http') else urljoin(self.config.api_url, url)
                    dl = await client.get(target_url)
                    dl.raise_for_status()
                    audio_bytes = dl.content
                elif mode == 'json_file':
                    file_path = str(_extract_json_key_path(data, self.config.response_key) or '').strip()
                    if not file_path:
                        return None
                    p = Path(file_path)
                    if not p.exists():
                        raise FileNotFoundError(f"TTS 返回文件不存在: {file_path}")
                    audio_bytes = p.read_bytes()
                elif mode == 'json_base64':
                    b64 = str(_extract_json_key_path(data, self.config.response_key) or '').strip()
                    audio_bytes = base64.b64decode(b64) if b64 else b''
                else:
                    raise ValueError(f"不支持的 TTS response_mode: {mode}")

        elapsed = int((time.time() - start) * 1000)
        self.logger.info(f"HTTP TTS 完成: {elapsed}ms, bytes={len(audio_bytes)}")
        return audio_bytes
