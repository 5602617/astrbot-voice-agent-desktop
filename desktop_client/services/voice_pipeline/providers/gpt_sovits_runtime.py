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

        self._last_health_check_ts: float = 0.0
        self._last_health_result: bool = False
        self._health_check_cache_ttl: float = 2.0  # 秒，可按需调大到 3~5
        self._health_probe_logged: bool = False

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.config.gpt_sovits_port}"

    async def ensure_started(self) -> None:
        if await asyncio.to_thread(self._is_service_ready):
            self.logger.info("检测到 GPT-SoVITS 服务已可用，直接复用: %s", self.base_url)
            return

        await asyncio.to_thread(self._start_process)

        deadline = time.time() + max(5, int(self.config.gpt_sovits_startup_timeout or 40))
        while time.time() < deadline:
            if await asyncio.to_thread(self._is_service_ready):
                self.logger.info("GPT-SoVITS 本地推理服务已就绪: %s", self.base_url)
                return
            await asyncio.sleep(0.4)

        raise RuntimeError("GPT-SoVITS 启动超时，服务仍未就绪")

    def _is_service_ready(self) -> bool:
        import socket

        host = "127.0.0.1"
        port = int(self.config.gpt_sovits_port)
        timeout = max(2, int(getattr(self.config, "gpt_sovits_health_timeout", 5) or 5))

        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            return False
    def _resolve_root_dir(self) -> Path:
        root = (getattr(self.config, "gpt_sovits_root_dir", "") or "").strip()
        if not root:
            raise RuntimeError("未配置 gpt_sovits_root_dir")
        root_path = Path(root).expanduser().resolve()
        if not root_path.exists():
            raise RuntimeError(f"GPT-SoVITS 根目录不存在: {root_path}")
        return root_path

    def _resolve_runtime_paths(self) -> tuple[str, str]:
        root = self._resolve_root_dir()
        script = root / "api_v2.py"
        if not script.exists():
            raise RuntimeError(f"未找到 api_v2.py: {script}")
        return str(script), str(root)

    def _build_runtime_env(self, workdir: str) -> dict:
        import os
        import site
        from pathlib import Path

        root = Path(workdir).resolve()
        env = os.environ.copy()

        # 对齐 webui.py 的基础环境
        env["version"] = "v2Pro"
        env["no_proxy"] = "localhost, 127.0.0.1, ::1"
        env["all_proxy"] = ""
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

        # TEMP 目录
        temp_dir = root / "TEMP"
        temp_dir.mkdir(exist_ok=True)
        env["TEMP"] = str(temp_dir)

        # 直接补 PYTHONPATH，避免只靠 cwd
        extra_paths = [
            str(root),
            str(root / "GPT_SoVITS"),
            str(root / "GPT_SoVITS" / "BigVGAN"),
            str(root / "tools"),
            str(root / "tools" / "asr"),
            str(root / "tools" / "uvr5"),
        ]

        old_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(extra_paths + ([old_pythonpath] if old_pythonpath else []))

        # 尽量复刻 webui.py 对 users.pth 的处理
        site_packages_roots = []
        try:
            for path in site.getsitepackages():
                if "packages" in path:
                    site_packages_roots.append(path)
        except Exception:
            pass

        if not site_packages_roots:
            site_packages_roots = [str(root / "runtime" / "Lib" / "site-packages")]

        for site_packages_root in site_packages_roots:
            p = Path(site_packages_root)
            if p.exists():
                try:
                    users_pth = p / "users.pth"
                    users_pth.write_text(
                        "\n".join(
                            [
                                str(root),
                                str(root / "GPT_SoVITS" / "BigVGAN"),
                                str(root / "tools"),
                                str(root / "tools" / "asr"),
                                str(root / "GPT_SoVITS"),
                                str(root / "tools" / "uvr5"),
                            ]
                        ),
                        encoding="utf-8",
                    )
                    self.logger.info("已写入 GPT-SoVITS users.pth: %s", users_pth)
                    break
                except Exception as e:
                    self.logger.warning("写入 users.pth 失败: %s", e)
        self.logger.info("GPT-SoVITS 启动环境 python_path=%s", self.config.gpt_sovits_python_path)
        self.logger.info("GPT-SoVITS 启动环境 cwd=%s", workdir)
        self.logger.info("GPT-SoVITS 启动环境 TEMP=%s", env.get("TEMP"))
        self.logger.info("GPT-SoVITS 启动环境 PYTHONPATH=%s", env.get("PYTHONPATH", ""))
        self.logger.info("GPT-SoVITS 启动环境 PATH(前500)=%s", env.get("PATH", "")[:500])

        return env

    def _start_process(self) -> None:
        if self._proc and self._proc.poll() is None:
            return

        script, workdir = self._resolve_runtime_paths()
        python_path = (self.config.gpt_sovits_python_path or "python").strip()
        env = self._build_runtime_env(workdir)

        cmd = [
            python_path,
            script,
            "-a",
            "127.0.0.1",
            "-p",
            str(self.config.gpt_sovits_port),
        ]

        self.logger.info("启动 GPT-SoVITS 本地推理子进程: %s", " ".join(cmd))
        self.logger.info("GPT-SoVITS 工作目录: %s", workdir)
        self.logger.info("GPT-SoVITS PYTHONPATH: %s", env.get("PYTHONPATH", ""))

        self._proc = subprocess.Popen(
            cmd,
            cwd=workdir,
            env=env,
            stdout=None,
            stderr=None,
        )

    async def is_healthy(self) -> bool:
        import time

        now = time.monotonic()
        if (now - self._last_health_check_ts) < self._health_check_cache_ttl:
            return self._last_health_result

        timeout = max(1, int(self.config.gpt_sovits_health_timeout or 2))

        def _looks_healthy(resp: httpx.Response, path: str) -> bool:
            ctype = (resp.headers.get("content-type") or "").lower()

            if path == "/health":
                return resp.status_code == 200

            if path == "/openapi.json":
                return resp.status_code == 200 and "application/json" in ctype

            if path == "/docs":
                return resp.status_code == 200 and "text/html" in ctype

            if path == "/":
                return resp.status_code in (200, 404)

            return False

        async with httpx.AsyncClient(timeout=timeout) as client:
            # 先走已探测成功的端点
            if self._detected_health_path:
                try:
                    r = await client.get(f"{self.base_url}{self._detected_health_path}")
                    healthy = _looks_healthy(r, self._detected_health_path)
                    self._last_health_check_ts = now
                    self._last_health_result = healthy
                    if healthy:
                        return True
                except Exception:
                    pass

                # 已缓存端点失效，清掉重新探测
                self._detected_health_path = None

            # 只在首次探测时按顺序尝试；优先用配置项，其次 /health，再 /openapi.json
            candidate_paths: list[str] = []
            configured = (self.config.gpt_sovits_health_endpoint or "").strip()

            for path in [configured, "/health", "/openapi.json", "/docs", "/"]:
                if path and path not in candidate_paths:
                    candidate_paths.append(path)

            for path in candidate_paths:
                try:
                    r = await client.get(f"{self.base_url}{path}")
                    if _looks_healthy(r, path):
                        self._detected_health_path = path
                        self._last_health_check_ts = now
                        self._last_health_result = True

                        if not self._health_probe_logged:
                            self.logger.info("GPT-SoVITS 健康检查端点自动探测为: %s", path)
                            self._health_probe_logged = True

                        return True
                except Exception:
                    continue

        self._last_health_check_ts = now
        self._last_health_result = False
        return False

    async def get_status(self) -> dict:
        """
        返回 GPT-SoVITS 当前状态：
        {
            "running": bool,           # 接口健康可用
            "process_alive": bool,     # 当前记录的子进程是否还活着
            "pid": int | None,
            "health_path": str | None,
            "base_url": str,
            "detail": str,
        }
        """
        proc_alive = self._proc is not None and self._proc.poll() is None
        pid = self._proc.pid if proc_alive and self._proc else None

        try:
            healthy = await self.is_healthy()
        except Exception as e:
            return {
                "running": False,
                "process_alive": proc_alive,
                "pid": pid,
                "health_path": self._detected_health_path,
                "base_url": self.base_url,
                "detail": f"健康检查异常: {e}",
            }

        if healthy:
            return {
                "running": True,
                "process_alive": proc_alive,
                "pid": pid,
                "health_path": self._detected_health_path,
                "base_url": self.base_url,
                "detail": "服务可用",
            }

        if proc_alive:
            return {
                "running": False,
                "process_alive": True,
                "pid": pid,
                "health_path": self._detected_health_path,
                "base_url": self.base_url,
                "detail": "子进程存在，但接口未就绪",
            }

        return {
            "running": False,
            "process_alive": False,
            "pid": None,
            "health_path": self._detected_health_path,
            "base_url": self.base_url,
            "detail": "服务未运行",
        }

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
        self._segment_callback = None
        self._active_session_tasks: dict[str, asyncio.Task] = {}
        self._session_generation: dict[str, int] = {}

    def set_segment_callback(self, callback):
        self._segment_callback = callback

    def _current_generation(self, session_id: str) -> int:
        return self._session_generation.get(session_id, 0)

    def _bump_generation(self, session_id: str) -> int:
        new_gen = self._current_generation(session_id) + 1
        self._session_generation[session_id] = new_gen
        return new_gen

    def _register_current_session_task(self, session_id: str | None) -> tuple[asyncio.Task | None, int]:
        if not session_id:
            return None, 0

        current_task = asyncio.current_task()
        if current_task is None:
            return None, self._current_generation(session_id)

        previous = self._active_session_tasks.get(session_id)
        if previous is not None and previous is not current_task and not previous.done():
            self.logger.info("GPT-SoVITS 新任务启动，取消旧推理任务: session=%s", session_id)
            previous.cancel()

        self._active_session_tasks[session_id] = current_task
        return current_task, self._current_generation(session_id)

    def _clear_current_session_task(self, session_id: str | None, task: asyncio.Task | None) -> None:
        if not session_id or task is None:
            return
        if self._active_session_tasks.get(session_id) is task:
            self._active_session_tasks.pop(session_id, None)

    async def warmup(self) -> None:
        await self._manager.ensure_started()
        await self._manager.sync_weights()

    async def get_status(self) -> dict:
        return await self._manager.get_status()

    async def synthesize_to_file(self, text: str, output_path: str | None = None, **kwargs) -> str | None:
        if not text.strip():
            return None

        request_id = kwargs.get("request_id")
        session_id = kwargs.get("session_id")

        await self.warmup()
        self._seq += 1
        out = Path(output_path) if output_path else self.cache_dir / f"gpt_sovits_{self._seq:04d}.wav"

        current_task, bound_generation = self._register_current_session_task(session_id)

        payload = {
            "text": text,
            "text_lang": self.config.gpt_sovits_default_language,
            "ref_audio_path": self.config.gpt_sovits_reference_audio_path,
            "prompt_lang": self.config.gpt_sovits_default_language,
            "prompt_text": self.config.gpt_sovits_reference_text,
            "text_split_method": (
                    self.config.gpt_sovits_segmentation_mode or "cut5"
            ).strip()
            if (self.config.gpt_sovits_segmentation_mode or "").strip() in {"cut0", "cut1", "cut2", "cut3", "cut4",
                                                                            "cut5"}
            else "cut5",
            "batch_size": 20,
            "batch_threshold": 0.75,
            "split_bucket": False,
            "parallel_infer": True,
            "sample_steps": 32,
            "top_k": 5,
            "top_p": 1.0,
            "temperature": 1.0,
            "repetition_penalty": 1.35,
            "speed_factor": 1.0,
            "fragment_interval": 0.3,
            "seed": -1,
            "media_type": "wav",
            "streaming_mode": False,
            "sentence_stream": True,
        }

        url = f"{self._manager.base_url}{self.config.gpt_sovits_tts_endpoint or '/tts'}"
        self.logger.info(
            "GPT-SoVITS TTS请求: url=%s session_id=%s request_id=%s payload=%s",
            url,
            session_id,
            request_id,
            payload,
        )

        try:
            timeout = max(10, int(self.config.gpt_sovits_request_timeout or 90))
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream("POST", url, json=payload) as r:
                    self.logger.info(
                        "GPT-SoVITS TTS响应: status=%s content-type=%s",
                        r.status_code,
                        r.headers.get("content-type"),
                    )

                    if r.status_code >= 400:
                        body = await r.aread()
                        raise RuntimeError(
                            f"GPT-SoVITS 推理失败: HTTP {r.status_code}, body={body[:500].decode(errors='ignore')}"
                        )

                    ctype = (r.headers.get("content-type") or "").lower()

                    def _ensure_not_stale() -> None:
                        if session_id and self._current_generation(session_id) != bound_generation:
                            self.logger.info(
                                "GPT-SoVITS 检测到过期句流任务，停止读取: session=%s request_id=%s generation=%s current=%s",
                                session_id,
                                request_id,
                                bound_generation,
                                self._current_generation(session_id),
                            )
                            raise asyncio.CancelledError()

                    if "audio" in ctype:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        first_chunk = True

                        async for chunk in r.aiter_bytes():
                            _ensure_not_stale()

                            if not chunk:
                                continue

                            mode = "wb" if first_chunk else "ab"
                            with open(out, mode) as f:
                                f.write(chunk)

                            first_chunk = False
                            self.logger.info("GPT-SoVITS 分段音频已写入: %s chunk_size=%s", out, len(chunk))

                        if out.exists():
                            self.logger.info("GPT-SoVITS 音频流接收完成: %s", out)
                            return str(out)

                        return None

                    if "application/octet-stream" in ctype:
                        out.parent.mkdir(parents=True, exist_ok=True)

                        buffer = bytearray()
                        waiting_meta = None
                        part_index = 0
                        part_paths: list[str] = []

                        def try_read_json_line(buf: bytearray):
                            pos = buf.find(b"\n")
                            if pos == -1:
                                return None
                            line = bytes(buf[:pos])
                            del buf[:pos + 1]
                            if not line.strip():
                                return None
                            return json.loads(line.decode("utf-8"))

                        async for chunk in r.aiter_bytes():
                            _ensure_not_stale()

                            if not chunk:
                                continue

                            buffer.extend(chunk)

                            while True:
                                if waiting_meta is None:
                                    meta = try_read_json_line(buffer)
                                    if meta is None:
                                        break

                                    self.logger.info("GPT-SoVITS 句级流 meta=%s", meta)

                                    if meta.get("type") == "end":
                                        waiting_meta = None
                                        break

                                    if meta.get("type") != "segment":
                                        raise RuntimeError(f"未知句级流消息: {meta}")

                                    waiting_meta = meta

                                if waiting_meta is not None:
                                    need = int(waiting_meta["bytes"])
                                    if len(buffer) < need:
                                        break

                                    audio_bytes = bytes(buffer[:need])
                                    del buffer[:need]

                                    _ensure_not_stale()

                                    part_index += 1
                                    part_path = out.with_name(f"{out.stem}_part{part_index:02d}{out.suffix}")
                                    part_path.write_bytes(audio_bytes)
                                    part_paths.append(str(part_path))

                                    self.logger.info(
                                        "GPT-SoVITS 句级音频已写入: %s bytes=%s text=%s",
                                        part_path,
                                        len(audio_bytes),
                                        waiting_meta.get("text", ""),
                                    )

                                    if self._segment_callback:
                                        try:
                                            callback_meta = dict(waiting_meta)
                                            if request_id:
                                                callback_meta["request_id"] = request_id
                                            if session_id:
                                                callback_meta["session_id"] = session_id

                                            self.logger.info(
                                                "GPT-SoVITS 准备触发句级回调: path=%s meta=%s callback=%r",
                                                part_path,
                                                callback_meta,
                                                self._segment_callback,
                                            )
                                            cb_result = self._segment_callback(str(part_path), callback_meta)
                                            if hasattr(cb_result, "__await__"):
                                                await cb_result
                                            self.logger.info("GPT-SoVITS 句级回调执行完成: %s", part_path)
                                        except Exception:
                                            self.logger.exception("GPT-SoVITS 句级回调执行失败: %s", part_path)

                                    waiting_meta = None

                        if part_paths:
                            self.logger.info("GPT-SoVITS 句级音频流接收完成: %s", part_paths)
                            return None

                        return None

                    body = await r.aread()
                    raise RuntimeError(
                        f"GPT-SoVITS 返回了未知 content-type: {ctype}, body={body[:200].decode(errors='ignore')}"
                    )

        except asyncio.CancelledError:
            self.logger.info(
                "GPT-SoVITS 推理任务被取消: session=%s request_id=%s",
                session_id,
                request_id,
            )
            raise
        finally:
            self._clear_current_session_task(session_id, current_task)
    async def synthesize_bytes(self, text: str, **kwargs) -> bytes | None:
        p = await self.synthesize_to_file(text)
        if not p:
            return None
        return Path(p).read_bytes()

    async def shutdown(self) -> None:
        self.stop()
        await self._manager.shutdown()

    def stop(self) -> None:
        for session_id in list(self._active_session_tasks.keys()):
            self.stop_session(session_id)

    def stop_session(self, session_id: str | None = None) -> None:
        if not session_id:
            for sid in list(self._active_session_tasks.keys()):
                self.stop_session(sid)
            return

        new_gen = self._bump_generation(session_id)
        task = self._active_session_tasks.get(session_id)

        if task is not None and not task.done():
            self.logger.info(
                "GPT-SoVITS 取消会话中的推理任务: session=%s generation=%s",
                session_id,
                new_gen,
            )
            task.cancel()
        else:
            self.logger.info(
                "GPT-SoVITS 会话轮次递增但当前无活跃推理任务: session=%s generation=%s",
                session_id,
                new_gen,
            )
