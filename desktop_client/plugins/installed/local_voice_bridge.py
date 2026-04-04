"""本地语音桥接插件。

将本地 ASR 文本接入云端 AstrBot，并在本地执行 TTS。
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import wave
from collections import deque
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

from desktop_client.plugins.base import IPlugin, PluginMetadata
from desktop_client.plugins.hooks import HookContext, HookResult, HookType
from desktop_client.services.local_voice_runtime import LocalVoiceRuntime

logger = logging.getLogger(__name__)


class _TTSSessionPlayer(QObject):
    def __init__(self, session_id: str, on_idle_callback=None, parent=None):
        super().__init__(parent)
        self.session_id = session_id
        self.on_idle_callback = on_idle_callback

        self.queue = deque()
        self.drain_task: asyncio.Task | None = None
        self.is_draining = False
        self.closed = False
        self.stream_ended = False
        self._seq = 0
        self._manual_stop = False
        self._current_item: dict | None = None
        self._current_audio_path: str | None = None

        self.audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)


    async def enqueue(self, audio_path: str, meta: dict | None = None):
        self._seq += 1
        item = {
            "seq": self._seq,
            "audio_path": str(audio_path),
            "meta": dict(meta or {}),
        }
        self.queue.append(item)

        logger.info(
            "TTS入队: session=%s seq=%s queue_size=%s path=%s text=%s is_last=%s",
            self.session_id,
            item["seq"],
            len(self.queue),
            item["audio_path"],
            item["meta"].get("text", ""),
            item["meta"].get("is_last"),
        )

        if not self.is_draining:
            self.is_draining = True
            self.drain_task = asyncio.create_task(self._drain(), name=f"tts_drain:{self.session_id}")

    async def _drain(self):
        try:
            while True:
                if not self.queue:
                    self.is_draining = False
                    self.drain_task = None

                    if self.on_idle_callback is not None:
                        try:
                            result = self.on_idle_callback(self.session_id)
                            if asyncio.iscoroutine(result):
                                await result
                        except Exception:
                            logger.exception("TTS idle 回调异常: session=%s", self.session_id)
                    return

                item = self.queue.popleft()
                await self._play_one(item)

        except asyncio.CancelledError:
            logger.info("TTS播放队列已取消: session=%s", self.session_id)
            raise

        finally:
            self.is_draining = False
            if asyncio.current_task() is self.drain_task:
                self.drain_task = None

    async def _play_one(self, item: dict):
        audio_path = item["audio_path"]
        meta = item["meta"]
        seq = item["seq"]
        self._current_item = item
        self._current_audio_path = str(audio_path)

        duration_sec = 3.0
        try:
            with contextlib.closing(wave.open(str(audio_path), "rb")) as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    duration_sec = frames / float(rate)
        except Exception:
            logger.exception("TTS播放器读取音频时长失败: %s", audio_path)

        finished = asyncio.get_running_loop().create_future()
        playback_started = False
        natural_end = False
        self._manual_stop = False

        def try_finish(reason: str):
            if not finished.done():
                finished.set_result(reason)

        def on_status_changed(status):
            nonlocal natural_end
            try:
                logger.info(
                    "TTS状态变化: session=%s seq=%s status=%s path=%s",
                    self.session_id, seq, status, audio_path
                )
                if (
                    status == QMediaPlayer.MediaStatus.EndOfMedia
                    and playback_started
                    and not self._manual_stop
                ):
                    natural_end = True
                    try_finish("end_of_media")
            except Exception:
                logger.exception("on_status_changed异常")

        def on_state_changed(state):
            nonlocal playback_started
            try:
                logger.info(
                    "TTS播放状态: session=%s seq=%s state=%s path=%s",
                    self.session_id, seq, state, audio_path
                )

                if state == QMediaPlayer.PlaybackState.PlayingState:
                    playback_started = True
                    return

                if state == QMediaPlayer.PlaybackState.StoppedState:
                    if self._manual_stop:
                        try_finish("manual_stop")
            except Exception:
                logger.exception("on_state_changed异常")

        def on_error_changed():
            try:
                err = self.player.error()
                err_str = self.player.errorString()
                if err:
                    logger.warning(
                        "TTS播放器错误: session=%s seq=%s err=%s err_str=%s path=%s",
                        self.session_id, seq, err, err_str, audio_path
                    )
                    try_finish(f"error:{err_str}")
            except Exception:
                logger.exception("on_error_changed异常")

        self.player.mediaStatusChanged.connect(on_status_changed)
        self.player.playbackStateChanged.connect(on_state_changed)
        self.player.errorChanged.connect(on_error_changed)

        try:
            logger.info(
                "TTS开始播放: session=%s seq=%s path=%s text=%s",
                self.session_id, seq, audio_path, meta.get("text", "")
            )

            self.player.setSource(QUrl.fromLocalFile(str(audio_path)))
            self.player.play()

            timeout = max(3.0, duration_sec + 2.0)

            try:
                reason = await asyncio.wait_for(finished, timeout=timeout)
                logger.info(
                    "TTS单句完成: session=%s seq=%s reason=%s started=%s natural_end=%s path=%s",
                    self.session_id, seq, reason, playback_started, natural_end, audio_path
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "TTS等待结束超时: session=%s seq=%s timeout=%.2f started=%s path=%s",
                    self.session_id, seq, timeout, playback_started, audio_path
                )
                self._manual_stop = True
                self.player.stop()

            await asyncio.sleep(0.08)
        finally:
            try:
                self.player.mediaStatusChanged.disconnect(on_status_changed)
            except Exception:
                pass
            try:
                self.player.playbackStateChanged.disconnect(on_state_changed)
            except Exception:
                pass
            try:
                self.player.errorChanged.disconnect(on_error_changed)
            except Exception:
                pass

            self._current_item = None
            self._current_audio_path = None

    def _safe_delete_file(self, path_str: str | None) -> None:
        if not path_str:
            return
        try:
            path = Path(path_str)
            if path.exists():
                path.unlink(missing_ok=True)
                logger.info("TTS音频文件已删除: session=%s path=%s", self.session_id, path)
        except Exception:
            logger.exception("删除TTS音频文件失败: session=%s path=%s", self.session_id, path_str)

    async def interrupt_and_cleanup(
            self,
            grace_seconds: float = 0.05,
            delete_files: bool = True,
            await_drain: bool = False,
            cancel_timeout: float = 0.25,
    ) -> None:
        """
        快速中断当前播放并清理队列。
        不再依赖 asyncio.Lock，避免 cancel 后锁卡死。
        """
        self._manual_stop = True

        if grace_seconds > 0:
            await asyncio.sleep(grace_seconds)

        current_path = self._current_audio_path
        queued_paths = [str(item.get("audio_path", "")) for item in self.queue]

        self.queue.clear()
        self.stream_ended = True

        try:
            self.player.stop()
        except Exception:
            logger.exception("停止TTS播放器失败: session=%s", self.session_id)

        task = self.drain_task
        if task and not task.done():
            task.cancel()

            if await_drain:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=cancel_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "等待TTS drain_task退出超时，转后台回收: session=%s timeout=%.2f",
                        self.session_id,
                        cancel_timeout,
                    )
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("等待TTS drain_task退出失败: session=%s", self.session_id)

        self.is_draining = False
        self.drain_task = None

        if delete_files:
            if current_path:
                self._safe_delete_file(current_path)
            for path in queued_paths:
                self._safe_delete_file(path)

        logger.info(
            "TTS已快速中断并清理: session=%s current=%s queued=%s await_drain=%s",
            self.session_id,
            current_path,
            len(queued_paths),
            await_drain,
        )

    async def stop_and_close(
            self,
            await_drain: bool = False,
            cancel_timeout: float = 0.20,
    ):
        self.closed = True
        self._manual_stop = True
        self.stream_ended = True

        queued_paths = [str(item.get("audio_path", "")) for item in self.queue]
        current_path = self._current_audio_path
        self.queue.clear()

        try:
            self.player.stop()
        except Exception:
            pass

        task = self.drain_task
        if task and not task.done():
            task.cancel()

            if await_drain:
                try:
                    await asyncio.wait_for(asyncio.shield(task), timeout=cancel_timeout)
                except asyncio.TimeoutError:
                    logger.warning(
                        "stop_and_close 等待 drain_task 超时: session=%s timeout=%.2f",
                        self.session_id,
                        cancel_timeout,
                    )
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.exception("stop_and_close 等待 drain_task 失败: session=%s", self.session_id)

        self.is_draining = False
        self.drain_task = None

        if current_path:
            self._safe_delete_file(current_path)
        for path in queued_paths:
            self._safe_delete_file(path)

        try:
            self.player.deleteLater()
        except Exception:
            pass
        try:
            self.audio_output.deleteLater()
        except Exception:
            pass


class LocalVoiceBridgePlugin(IPlugin):
    """ASR -> 云端 AstrBot -> 本地 TTS 的桥接插件。"""

    def __init__(self):
        super().__init__()
        self._session_players: dict[str, _TTSSessionPlayer] = {}
        self._current_tts_session_id = ""
        self._tts_request_session_map: dict[str, str] = {}
        self._runtime = None
        self._app = None
        self._shutdown_task: asyncio.Task | None = None
        self._session_generation: dict[str, int] = {}
        self._request_generation: dict[tuple[str, str], int] = {}

    def _get_or_create_session_player(self, session_id: str) -> _TTSSessionPlayer:
        player = self._session_players.get(session_id)
        if player is None:
            player = _TTSSessionPlayer(
                session_id=session_id,
                on_idle_callback=self._cleanup_session_player_if_finished,
            )
            self._session_players[session_id] = player
        return player

    def _bump_session_generation(self, session_id: str) -> int:
        new_gen = self._session_generation.get(session_id, 0) + 1
        self._session_generation[session_id] = new_gen
        return new_gen

    def _current_generation(self, session_id: str) -> int:
        return self._session_generation.get(session_id, 0)

    def _bind_request_generation(self, session_id: str, request_id: str | None) -> int:
        if not request_id:
            return self._current_generation(session_id)
        key = (session_id, request_id)
        if key not in self._request_generation:
            self._request_generation[key] = self._current_generation(session_id)
        return self._request_generation[key]

    def _is_request_stale(self, session_id: str, request_id: str | None) -> bool:
        if not request_id:
            return False
        req_gen = self._bind_request_generation(session_id, request_id)
        return req_gen != self._current_generation(session_id)

    async def _mark_session_tts_end(self, session_id: str) -> None:
        player = self._session_players.get(session_id)
        if not player:
            return

        player.stream_ended = True
        logger.info("LocalVoiceBridge 收到最后一句标记: session=%s", session_id)

    async def _cleanup_session_player_if_finished(self, session_id: str) -> None:
        player = self._session_players.get(session_id)
        if not player:
            return

        if player.stream_ended and (not player.is_draining) and (not player.queue):
            await player.stop_and_close()
            self._session_players.pop(session_id, None)
            logger.info("LocalVoiceBridge 会话播放器已销毁: session=%s", session_id)

    async def _interrupt_session_tts(
            self,
            session_id: str,
            grace_seconds: float = 0.05,
            await_close: bool = False,
    ) -> None:
        """
        快速打断当前会话 TTS。

        默认策略：
        - 新输入到来时只做“快中断”，不等待彻底关闭
        - 避免 PRE_MESSAGE_SEND 被旧 TTS 清理过程阻塞
        - 真正关闭放到后台或 shutdown 流程
        """

        async def _interrupt_one(sid: str, player: _TTSSessionPlayer):
            await player.interrupt_and_cleanup(
                grace_seconds=grace_seconds,
                delete_files=True,
                await_drain=False,
            )

            if await_close:
                await player.stop_and_close(await_drain=False)
                self._session_players.pop(sid, None)
                logger.info("LocalVoiceBridge 已关闭会话TTS播放器: session=%s", sid)
            else:
                logger.info("LocalVoiceBridge 已快速打断会话TTS: session=%s", sid)

        if session_id:
            player = self._session_players.get(session_id)
            if player:
                await _interrupt_one(session_id, player)
            return

        for sid, player in list(self._session_players.items()):
            await _interrupt_one(sid, player)

    async def _on_tts_segment_ready(self, audio_path: str, meta: dict) -> None:
        logger.info("LocalVoiceBridge 句段TTS回调: path=%s meta=%s", audio_path, meta)

        meta = meta or {}
        request_id = meta.get("request_id")
        session_id = meta.get("session_id")

        if not session_id and request_id:
            session_id = self._tts_request_session_map.get(request_id)

        if not session_id:
            session_id = self._current_tts_session_id or "default_tts_session"

        if self._is_request_stale(session_id, request_id):
            logger.info(
                "LocalVoiceBridge 丢弃过期TTS句段: session=%s request_id=%s path=%s",
                session_id,
                request_id,
                audio_path,
            )
            try:
                Path(audio_path).unlink(missing_ok=True)
            except Exception:
                logger.exception("删除过期TTS句段失败: %s", audio_path)
            return

        player = self._get_or_create_session_player(session_id)

        meta_type = meta.get("type")
        is_last = bool(meta.get("is_last"))

        if meta_type != "segment":
            logger.info(
                "LocalVoiceBridge 忽略未知TTS meta类型: session=%s type=%s meta=%s",
                session_id,
                meta_type,
                meta,
            )
            return

        await player.enqueue(audio_path, meta)

        if is_last:
            await self._mark_session_tts_end(session_id)
            await self._cleanup_session_player_if_finished(session_id)

            if request_id:
                self._tts_request_session_map.pop(request_id, None)
                self._request_generation.pop((session_id, request_id), None)

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="local_voice_bridge",
            version="0.1.0",
            author="AstrBot Desktop",
            description="本地 ASR 文本转发到云端 AstrBot，并在本地执行 TTS 播放",
            tags=["voice", "asr", "tts", "cloud-adapter"],
        )

    def on_load(self) -> bool:
        bridge = getattr(self._manager, "app_bridge", None)
        config = getattr(self._manager, "app_config", None)
        app = getattr(self._manager, "app_instance", None)
        if bridge is None or config is None:
            logger.error("LocalVoiceBridgePlugin: 缺少 bridge/config 上下文")
            return False

        self._app = app
        self._runtime = LocalVoiceRuntime(bridge=bridge, config=config)
        self._runtime.set_audio_generated_callback(self._on_audio_generated)
        self._runtime.set_tts_segment_callback(self._on_tts_segment_ready)
        self._seen_complete_keys: set[str] = set()

        self.register_hook(HookType.PRE_MESSAGE_SEND, self._on_pre_message_send)
        self.register_hook(HookType.POST_MESSAGE_RECEIVE, self._on_post_message_receive)
        self.load_config()

        if "tts_enabled" in self.config:
            self._runtime.set_tts_enabled(bool(self.config.get("tts_enabled", True)))

        return True

    def on_unload(self) -> None:
        self.set_config_value("tts_enabled", self._runtime.tts_enabled)
        self.save_config()
        super().on_unload()

    def on_enable(self) -> bool:
        if getattr(self, "_start_task", None) and not self._start_task.done():
            logger.info("LocalVoiceBridge 启动任务已存在，跳过重复创建")
            return True
        self._start_task = asyncio.create_task(self._runtime.start())
        return True

    async def shutdown_async(self) -> None:
        # 先取消启动任务
        if getattr(self, "_start_task", None) and not self._start_task.done():
            self._start_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._start_task

        # 退出时只做“快速中断 + 非阻塞关闭”
        try:
            await self._interrupt_session_tts(
                session_id="",
                grace_seconds=0.0,
                await_close=False,
            )
        except Exception:
            logger.exception("关闭阶段快速中断 TTS 失败")

        # 尝试把播放器对象 deleteLater 掉，但不长时间等 drain_task
        for sid, player in list(self._session_players.items()):
            try:
                await player.stop_and_close(await_drain=False)
            except Exception:
                logger.exception("关闭会话播放器失败: session=%s", sid)
            finally:
                self._session_players.pop(sid, None)

        if self._runtime is not None:
            try:
                await self._runtime.shutdown_async()
            except Exception:
                logger.exception("关闭本地语音运行时失败")

    def on_disable(self) -> None:
        if self._shutdown_task and not self._shutdown_task.done():
            return

        try:
            self._shutdown_task = asyncio.create_task(self.shutdown_async())
        except RuntimeError:
            logger.debug("事件循环已关闭，跳过 LocalVoiceBridge 异步关闭任务")
        except Exception:
            logger.exception("创建 LocalVoiceBridge 关闭任务失败")

    async def _on_audio_generated(self, session_id: str, audio_path: str) -> None:
        logger.info("LocalVoiceBridge 整段TTS完成(不参与播放队列): session=%s path=%s", session_id, audio_path)

    async def _on_post_message_receive(self, context: HookContext) -> HookResult:
        msg_type = context.get("msg_type", "")
        session_id = context.get("session_id", "")
        metadata = context.get("metadata", {}) or {}
        request_id = metadata.get("request_id")

        if msg_type == "text":
            self._bind_request_generation(session_id, request_id)
            if self._is_request_stale(session_id, request_id):
                logger.info(
                    "LocalVoiceBridge 忽略过期文本事件: session=%s request_id=%s",
                    session_id,
                    request_id,
                )
                return HookResult.CONTINUE
            self._runtime.on_reply_chunk(
                session_id=session_id,
                message=context.get("message", ""),
                streaming=bool(context.get("streaming", False)),
                metadata=metadata,
            )

        elif msg_type == "complete":
            self._bind_request_generation(session_id, request_id)
            if self._is_request_stale(session_id, request_id):
                logger.info(
                    "LocalVoiceBridge 忽略过期complete事件: session=%s request_id=%s",
                    session_id,
                    request_id,
                )
                return HookResult.CONTINUE

            request_id_key = request_id or "default"
            key = f"{session_id}:{request_id_key}"

            if key in self._seen_complete_keys:
                logger.info("LocalVoiceBridge TTS去重命中: key=%s", key)
                return HookResult.CONTINUE

            self._seen_complete_keys.add(key)
            if len(self._seen_complete_keys) > 2048:
                self._seen_complete_keys = set(list(self._seen_complete_keys)[-512:])

            self._current_tts_session_id = session_id
            if request_id:
                self._tts_request_session_map[request_id] = session_id

            self._runtime.on_reply_end(session_id=session_id, metadata=metadata)
            logger.info(
                "LocalVoiceBridge TTS触发: msg_type=complete request_id=%s session_id=%s generation=%s",
                request_id,
                session_id,
                self._current_generation(session_id),
            )

        elif msg_type == "end":
            logger.debug("LocalVoiceBridge 忽略 end 事件（仅 complete 触发 TTS）")

        return HookResult.CONTINUE

    async def _on_pre_message_send(self, context: HookContext) -> HookResult:
        """
        新输入到来时：
        1. 先递增 generation，软取消旧轮次
        2. 立即中断 runtime 里的当前 TTS / reply_end
        3. 快速停止本地播放器，但不等待彻底关闭
        4. 立刻放行，让第二轮文本先发出去
        """
        session_id = context.get("session_id", "")
        new_gen = self._bump_session_generation(session_id)

        logger.info(
            "LocalVoiceBridge 新输入到来，轮次递增: session=%s generation=%s",
            session_id,
            new_gen,
        )

        # 先中断 pipeline / provider / reply_end 任务
        self._runtime.interrupt_current_turn(session_id=session_id, reason="new_input")

        # 再快速停止本地播放，但不要阻塞新的文本发送
        try:
            await self._interrupt_session_tts(
                session_id=session_id,
                grace_seconds=0.03,
                await_close=False,
            )
        except Exception:
            logger.exception("LocalVoiceBridge 快速中断会话TTS失败: session=%s", session_id)

        return HookResult.CONTINUE

    async def submit_asr_text(self, text: str, session_id: Optional[str] = None) -> bool:
        return await self._runtime.submit_asr_text(text=text, session_id=session_id)

    async def submit_asr_audio_file(
        self, audio_path: str, session_id: Optional[str] = None
    ) -> Optional[str]:
        return await self._runtime.handle_audio_file(audio_path, session_id=session_id)

    async def submit_asr_audio_bytes(
        self,
        audio_bytes: bytes,
        sample_rate: Optional[int] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        return await self._runtime.handle_audio_bytes(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            session_id=session_id,
        )

    def set_tts_enabled(self, enabled: bool) -> None:
        self._runtime.set_tts_enabled(enabled)
        self.set_config_value("tts_enabled", enabled)
        self.save_config()

    async def start_gpt_sovits(self) -> bool:
        if self._runtime is None:
            return False
        return await self._runtime.start_gpt_sovits()

    async def stop_gpt_sovits(self, force: bool = False) -> bool:
        if self._runtime is None:
            return True
        return await self._runtime.stop_gpt_sovits(force=force)

    async def reload_from_config(self) -> None:
        await self._runtime.reload_from_config()

    async def get_gpt_sovits_status(self) -> dict:
        if self._runtime is None:
            return {
                "running": False,
                "process_alive": False,
                "pid": None,
                "health_path": None,
                "base_url": "",
                "detail": "runtime 未初始化",
            }
        return await self._runtime.get_gpt_sovits_status()
