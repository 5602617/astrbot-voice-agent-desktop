"""
远程命令处理器

处理服务端通过 WebSocket 下发的命令，如截图等。
"""
import asyncio
import base64
import logging
import time
from io import BytesIO
from typing import TYPE_CHECKING, Optional, Dict, Any, Callable, Tuple

from PySide6.QtCore import QObject, Signal

if TYPE_CHECKING:
    from ..config import ClientConfig
    from ..bridge import MessageBridge

logger = logging.getLogger(__name__)


class RemoteCommandHandler(QObject):
    """
    远程命令处理器

    处理服务端下发的命令，如：
    - screenshot: 截图并返回 base64 编码的图片
    """

    # 信号定义
    command_received = Signal(str, str, dict)  # command, request_id, params
    command_completed = Signal(
        str, str, bool, str
    )  # command, request_id, success, message

    def __init__(
        self,
        config: "ClientConfig",
        bridge: Optional["MessageBridge"] = None,
        parent: Optional[QObject] = None,
    ):
        """
        初始化远程命令处理器

        Args:
            config: 客户端配置
            bridge: 消息桥接器（用于访问 WebSocket 客户端）
            parent: 父对象
        """
        super().__init__(parent)
        self._config = config
        self._bridge = bridge
        self._floating_ball = None

        # 命令处理器映射
        self._command_handlers: Dict[str, Callable] = {
            "screenshot": self._handle_screenshot_command,
        }

    def set_floating_ball(self, floating_ball: Any) -> None:
        """设置悬浮球实例（用于隐藏/显示窗口）"""
        self._floating_ball = floating_ball

    def set_bridge(self, bridge: "MessageBridge") -> None:
        """设置消息桥接器（用于访问 WebSocket 客户端）"""
        self._bridge = bridge

    async def _set_busy_state(
        self, is_busy: bool, operation: str = "", duration: int = 60
    ) -> None:
        """
        设置忙碌状态，通知服务端延长超时时间

        Args:
            is_busy: 是否进入忙碌状态
            operation: 操作名称
            duration: 预计操作持续时间（秒）
        """
        try:
            if self._bridge and self._bridge.api_client.ws_client:
                ws_client = self._bridge.api_client.ws_client
                if ws_client.is_connected:
                    await ws_client.set_busy_state(is_busy, operation, duration)
                else:
                    logger.warning("WebSocket 未连接，无法报告忙碌状态")
            else:
                logger.warning("Bridge 或 WebSocket 客户端未设置，无法报告忙碌状态")
        except Exception as e:
            logger.error(f"设置忙碌状态失败: {e}")

    async def handle_command(
        self, command: str, request_id: str, params: dict
    ) -> Dict[str, Any]:
        """
        处理远程命令

        Args:
            command: 命令名称
            request_id: 请求 ID
            params: 命令参数

        Returns:
            命令执行结果字典
        """
        logger.info(f"处理远程命令: {command}, request_id={request_id}")
        self.command_received.emit(command, request_id, params)

        handler = self._command_handlers.get(command)

        if handler is None:
            error_msg = f"未知命令: {command}"
            logger.warning(error_msg)
            self.command_completed.emit(command, request_id, False, error_msg)
            return {"success": False, "error_message": error_msg}

        try:
            result = await handler(request_id, params)
            success = result.get("success", False)
            message = result.get("error_message", "") if not success else "成功"
            self.command_completed.emit(command, request_id, success, message)
            return result
        except Exception as e:
            error_msg = f"命令执行异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.command_completed.emit(command, request_id, False, error_msg)
            return {"success": False, "error_message": error_msg}

    def _build_screenshot_success_result(
            self, image_base64: str, width: int, height: int
    ) -> Dict[str, Any]:
        """按原版协议包装 screenshot 成功结果"""
        return {
            "success": True,
            "image_base64": image_base64,
            "width": width,
            "height": height,
            "timestamp": time.time(),
        }

    def _build_screenshot_error_result(self, error_message: str) -> Dict[str, Any]:
        """按原版协议包装 screenshot 失败结果"""
        return {
            "success": False,
            "error_message": error_message,
        }

    def _get_remote_screenshot_target_size(self) -> Tuple[int, int]:
        """获取远程截图压缩目标尺寸"""
        proactive = getattr(self._config, "proactive", None)

        width = getattr(proactive, "screenshot_width", 1600) if proactive else 1600
        height = getattr(proactive, "screenshot_height", 900) if proactive else 900

        try:
            width = int(width or 1600)
            height = int(height or 900)
        except Exception:
            width, height = 1600, 900

        return max(width, 640), max(height, 480)

    def _resize_image_for_remote(self, image):
        """将远程截图压缩到较合理尺寸，避免回包过大/过慢"""
        try:
            from PIL import Image

            target_w, target_h = self._get_remote_screenshot_target_size()
            orig_w, orig_h = image.size

            if orig_w <= target_w and orig_h <= target_h:
                return image, orig_w, orig_h

            resized = image.copy()

            if hasattr(Image, "Resampling"):
                resample = Image.Resampling.LANCZOS
            else:
                resample = Image.LANCZOS

            resized.thumbnail((target_w, target_h), resample)
            return resized, orig_w, orig_h
        except Exception:
            logger.exception("远程截图压缩失败，回退原图")
            w, h = image.size
            return image, w, h

    def _capture_remote_screenshot_sync(self, screenshot_type: str) -> Dict[str, Any]:
        """同步执行截图、压缩、编码，放到线程里跑，避免阻塞事件循环"""
        from ..services.screen_capture import ScreenCaptureService

        save_dir = self._config.storage.image_save_path or "./temp/screenshots"
        service = ScreenCaptureService(save_dir=save_dir)

        capture_started = time.perf_counter()

        if screenshot_type == "full":
            image = service.capture_full_screen()
        else:
            # 远程区域截图仍然回退成全屏
            image = service.capture_full_screen()

        capture_cost = time.perf_counter() - capture_started

        if image is None:
            return self._build_screenshot_error_result("截图失败：无法捕获屏幕")

        resized_image, orig_w, orig_h = self._resize_image_for_remote(image)
        final_w, final_h = resized_image.size

        encode_started = time.perf_counter()
        image_bytes = service.capture_to_bytes(resized_image)
        encode_cost = time.perf_counter() - encode_started

        if image_bytes is None:
            return self._build_screenshot_error_result("截图失败：无法编码图片")

        base64_started = time.perf_counter()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        base64_cost = time.perf_counter() - base64_started

        logger.info(
            "远程截图完成: original=%sx%s final=%sx%s png_bytes=%s base64_len=%s "
            "capture=%.3fs encode=%.3fs b64=%.3fs",
            orig_w,
            orig_h,
            final_w,
            final_h,
            len(image_bytes),
            len(image_base64),
            capture_cost,
            encode_cost,
            base64_cost,
        )

        result = self._build_screenshot_success_result(
            image_base64=image_base64,
            width=final_w,
            height=final_h,
        )
        result["original_width"] = orig_w
        result["original_height"] = orig_h
        return result

    async def _handle_screenshot_command(
        self, request_id: str, params: dict
    ) -> Dict[str, Any]:
        """
        处理截图命令

        Args:
            request_id: 请求 ID
            params: 命令参数
                - type: 截图类型 ("full" 或 "region")

        Returns:
            包含截图结果的字典
        """
        screenshot_type = params.get("type", "full")

        logger.info("执行远程截图: type=%s, request_id=%s", screenshot_type, request_id)

        await self._set_busy_state(True, "screenshot", 60)

        try:
            # 隐藏悬浮球，避免截进去
            if self._floating_ball:
                self._floating_ball.hide()

            await asyncio.sleep(0.2)

            started = time.perf_counter()

            # 放到线程里执行，避免主事件循环长时间阻塞
            result = await asyncio.to_thread(
                self._capture_remote_screenshot_sync,
                screenshot_type,
            )

            total_cost = time.perf_counter() - started

            logger.info(
                "远程截图命令结束: request_id=%s success=%s total=%.3fs",
                request_id,
                result.get("success", False),
                total_cost,
            )

            return result

        except ImportError as e:
            error_msg = f"截图服务不可用: {str(e)}"
            logger.error(error_msg)
            return self._build_screenshot_error_result(error_msg)
        except Exception as e:
            error_msg = f"截图异常: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._build_screenshot_error_result(error_msg)
        finally:
            if self._floating_ball:
                self._floating_ball.show()

            await self._set_busy_state(False, "screenshot")

    def register_command(self, command: str, handler: Callable) -> None:
        """
        注册自定义命令处理器

        Args:
            command: 命令名称
            handler: 处理函数，签名: async (request_id, params) -> dict
        """
        self._command_handlers[command] = handler
        logger.info(f"已注册远程命令处理器: {command}")

    def unregister_command(self, command: str) -> None:
        """
        注销命令处理器

        Args:
            command: 命令名称
        """
        if command in self._command_handlers:
            del self._command_handlers[command]
            logger.info(f"已注销远程命令处理器: {command}")

    @property
    def supported_commands(self) -> list:
        """获取支持的命令列表"""
        return list(self._command_handlers.keys())
