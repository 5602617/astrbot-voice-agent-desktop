from __future__ import annotations

import asyncio
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse


class SovitsApiManager:
    """管理本地 GPT-SoVITS API 子进程。"""

    def __init__(self, config, logger):
        self._config = config
        self._logger = logger
        self._process: subprocess.Popen | None = None

    def is_enabled(self) -> bool:
        v = self._config.voice
        return bool(
            getattr(v, "sovits_auto_start", False)
            and getattr(v, "tts_provider_type", "") == "runtime"
            and str(getattr(v, "tts_runtime_backend", "")).lower() in ("sovits", "gpt_sovits")
        )

    async def ensure_started(self) -> None:
        if not self.is_enabled():
            return

        host, port = self._resolve_host_port()
        if self._port_open(host, port):
            self._logger.info("检测到 SoVITS API 已运行: %s:%s", host, port)
            return

        cmd, cwd, env = self._build_command()
        self._logger.info("启动 SoVITS API: cwd=%s cmd=%s", cwd, shlex.join(cmd))
        self._process = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        ok = await self._wait_until_ready(host, port, timeout=float(getattr(self._config.voice, "sovits_start_timeout", 25)))
        if not ok:
            raise RuntimeError(f"SoVITS API 启动超时，未监听 {host}:{port}")

        self._logger.info("SoVITS API 启动成功: %s:%s", host, port)

    async def restart_if_needed(self) -> None:
        await self.stop()
        await self.ensure_started()

    async def stop(self) -> None:
        if self._process is None:
            return
        proc = self._process
        self._process = None
        if proc.poll() is not None:
            return

        self._logger.info("停止 SoVITS API 进程 pid=%s", proc.pid)
        proc.terminate()
        try:
            await asyncio.to_thread(proc.wait, 5)
        except Exception:
            proc.kill()

    def _resolve_host_port(self) -> tuple[str, int]:
        api_url = str(getattr(self._config.voice, "tts_api_url", "") or "").strip()
        parsed = urlparse(api_url) if api_url else None
        host = getattr(self._config.voice, "sovits_bind_host", "127.0.0.1")
        port = int(getattr(self._config.voice, "sovits_bind_port", 9880) or 9880)
        if parsed and parsed.hostname:
            host = parsed.hostname
            if parsed.port:
                port = parsed.port
        return host, port

    def _build_command(self) -> tuple[list[str], Path, dict[str, str]]:
        voice = self._config.voice
        project_dir = Path(getattr(voice, "sovits_project_dir", "") or "").expanduser().resolve()
        if not project_dir.exists():
            raise FileNotFoundError(f"SoVITS 项目目录不存在: {project_dir}")

        script = str(getattr(voice, "sovits_api_script", "api_v2.py") or "api_v2.py")
        script_path = Path(script)
        if not script_path.is_absolute():
            script_path = project_dir / script_path
        if not script_path.exists():
            raise FileNotFoundError(f"SoVITS API 脚本不存在: {script_path}")

        python_bin = str(getattr(voice, "sovits_python", "") or "").strip() or sys.executable
        host, port = self._resolve_host_port()
        cfg = str(getattr(voice, "sovits_tts_config", "") or "GPT_SoVITS/configs/tts_infer.yaml")

        cmd = [python_bin, str(script_path), "-a", host, "-p", str(port)]
        if cfg.strip():
            cmd.extend(["-c", cfg.strip()])

        env = os.environ.copy()
        env.setdefault("GPT_SOVITS_TTS_URL", str(getattr(voice, "tts_api_url", "") or f"http://{host}:{port}/tts"))
        ref = str(getattr(voice, "tts_ref_audio_path", "") or "")
        if ref:
            env.setdefault("TTS_REF_AUDIO_PATH", ref)
        ptxt = str(getattr(voice, "tts_prompt_text", "") or "")
        if ptxt:
            env.setdefault("TTS_PROMPT_TEXT", ptxt)
        plang = str(getattr(voice, "tts_prompt_lang", "") or "")
        if plang:
            env.setdefault("TTS_PROMPT_LANG", plang)
        tlang = str(getattr(voice, "tts_language", "") or "")
        if tlang:
            env.setdefault("TTS_TEXT_LANG", tlang)

        return cmd, project_dir, env

    async def _wait_until_ready(self, host: str, port: int, timeout: float) -> bool:
        deadline = asyncio.get_event_loop().time() + max(1.0, timeout)
        while asyncio.get_event_loop().time() < deadline:
            if self._port_open(host, port):
                return True
            await asyncio.sleep(0.3)
        return False

    @staticmethod
    def _port_open(host: str, port: int) -> bool:
        try:
            with socket.create_connection((host, port), timeout=0.6):
                return True
        except Exception:
            return False
