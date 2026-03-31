from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path
from typing import Optional

import httpx

from ..base import BaseTTSProvider
from ..models import TTSProviderConfig


class GPTSoVITSRuntimeManager:
    def __init__(self, config: TTSProviderConfig, logger):
        self.config = config
        self.logger = logger
        self._proc: Optional[subprocess.Popen] = None
        self._detected_health_path: Optional[str] = None
        self._weights_synced = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.gpt_sovits_port}"

    async def ensure_started(self) -> None:
        if await self.is_healthy():
            return
        await asyncio.to_thread(self._start_process)
        deadline = time.time() + max(5, int(self.config.gpt_sovits_startup_timeout or 40))
        while time.time() < deadline:
            if await self.is_healthy():
                self.logger.info("GPT-SoVITS 本地推理服务已就绪: %s", self.base_url)
                return
            await asyncio.sleep(0.4)
        raise RuntimeError("GPT-SoVITS 启动超时，健康检查未通过")

    def _start_process(self) -> None:
        if self._proc and self._proc.poll() is None:
            return
        script = (self.config.gpt_sovits_api_script_path or "").strip()
        if not script:
            raise RuntimeError("未配置 gpt_sovits_api_script_path")
        if not Path(script).exists():
            raise RuntimeError(f"GPT-SoVITS API脚本不存在: {script}")
        python_path = (self.config.gpt_sovits_python_path or "python").strip()
        workdir = (self.config.gpt_sovits_working_dir or str(Path(script).parent)).strip()
        if not Path(workdir).exists():
            raise RuntimeError(f"GPT-SoVITS 工作目录不存在: {workdir}")
        cmd = [python_path, script, "-a", "127.0.0.1", "-p", str(self.config.gpt_sovits_port)]
        cfg = (self.config.gpt_sovits_tts_config_path or "").strip()
        if cfg:
            if not Path(cfg).exists():
                raise RuntimeError(f"GPT-SoVITS tts_infer.yaml 不存在: {cfg}")
            cmd += ["-c", cfg]
        self.logger.info("启动 GPT-SoVITS 本地推理子进程: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    async def is_healthy(self) -> bool:
        timeout = max(1, int(self.config.gpt_sovits_health_timeout or 2))
        async with httpx.AsyncClient(timeout=timeout) as client:
            if self._detected_health_path:
                try:
                    r = await client.get(f"{self.base_url}{self._detected_health_path}")
                    return r.status_code < 500
                except Exception:
                    pass

            for path in [self.config.gpt_sovits_health_endpoint, "/health", "/docs", "/openapi.json"]:
                if not path:
                    continue
                try:
                    r = await client.get(f"{self.base_url}{path}")
                    if r.status_code < 500:
                        self._detected_health_path = path
                        self.logger.info("GPT-SoVITS 健康检查端点自动探测为: %s", path)
                        return True
                except Exception:
                    continue

            # api_v2.py 默认一定有 /control；用于兜底探测（不执行命令）
            try:
                r = await client.get(f"{self.base_url}/control")
                if r.status_code < 500:
                    self._detected_health_path = "/control"
                    return True
            except Exception:
                return False
        return False

    async def sync_weights(self) -> None:
        if self._weights_synced:
            return
        timeout = max(2, int(self.config.gpt_sovits_request_timeout or 90))
        async with httpx.AsyncClient(timeout=timeout) as client:
            gpt_w = (getattr(self.config, "gpt_sovits_selected_gpt_weights", "") or "").strip()
            sovits_w = (getattr(self.config, "gpt_sovits_selected_sovits_weights", "") or "").strip()
            if gpt_w:
                r = await client.get(f"{self.base_url}/set_gpt_weights", params={"weights_path": gpt_w})
                if r.status_code >= 400:
                    raise RuntimeError(f"设置 GPT 权重失败: HTTP {r.status_code}")
            if sovits_w:
                r = await client.get(f"{self.base_url}/set_sovits_weights", params={"weights_path": sovits_w})
                if r.status_code >= 400:
                    raise RuntimeError(f"设置 SoVITS 权重失败: HTTP {r.status_code}")
        self._weights_synced = True

    async def shutdown(self) -> None:
        if not bool(getattr(self.config, "gpt_sovits_auto_shutdown_on_exit", True)):
            return
        if self._proc and self._proc.poll() is None:
            self.logger.info("关闭 GPT-SoVITS 本地推理子进程")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._proc = None


class GPTSoVITSRuntimeProvider(BaseTTSProvider):
    def __init__(self, config: TTSProviderConfig, logger, cache_dir: str):
        self.config = config
        self.logger = logger
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._manager = GPTSoVITSRuntimeManager(config, logger)
        self._seq = 0

    async def warmup(self) -> None:
        await self._manager.ensure_started()
        await self._manager.sync_weights()

    async def synthesize_to_file(self, text: str, output_path: str | None = None, **kwargs) -> str | None:
        if not text.strip():
            return None
        await self.warmup()
        self._seq += 1
        out = Path(output_path) if output_path else self.cache_dir / f"gpt_sovits_{self._seq:04d}.wav"
        payload = {
            "text": text,
            "text_lang": self.config.gpt_sovits_default_language,
            "ref_audio_path": self.config.gpt_sovits_reference_audio_path,
            "prompt_lang": self.config.gpt_sovits_default_language,
            "prompt_text": self.config.gpt_sovits_reference_text,
            "text_split_method": self.config.gpt_sovits_segmentation_mode or "cut5",
            "media_type": "wav",
            "streaming_mode": False,
        }
        extra = json.loads(self.config.gpt_sovits_segmentation_params_json or "{}")
        if isinstance(extra, dict):
            payload.update(extra)
        url = f"{self._manager.base_url}{self.config.gpt_sovits_tts_endpoint or '/tts'}"
        async with httpx.AsyncClient(timeout=max(10, int(self.config.gpt_sovits_request_timeout or 90))) as client:
            r = await client.post(url, json=payload)
            if r.status_code >= 400:
                raise RuntimeError(f"GPT-SoVITS 推理失败: HTTP {r.status_code}")
            ctype = (r.headers.get("content-type") or "").lower()
            if "audio" in ctype:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(r.content)
        return str(out) if out.exists() else None

    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        p = await self.synthesize_to_file(text)
        if not p:
            return None
        return Path(p).read_bytes()

    async def shutdown(self) -> None:
        await self._manager.shutdown()

    def stop(self) -> None:
        pass
