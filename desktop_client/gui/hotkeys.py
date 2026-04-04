"""
全局快捷键系统。

实现说明：
- 前台快捷键：Qt 事件过滤器（应用激活时）
- 全局快捷键：pynput.keyboard.Listener（应用失焦/最小化时仍可触发）
- ASR：支持按下/松开的全局监听
"""

from __future__ import annotations

from typing import Dict, Optional, Callable
from dataclasses import dataclass
import logging

from PySide6.QtCore import QObject, Signal, QEvent
from PySide6.QtGui import QKeySequence, QShortcut, QKeyEvent
from PySide6.QtWidgets import QWidget, QApplication

logger = logging.getLogger(__name__)


@dataclass
class HotkeyConfig:
    """快捷键配置"""

    toggle_chat: str = "Ctrl+Shift+A"
    region_screenshot: str = "Ctrl+Shift+S"
    full_screenshot: str = "Ctrl+Shift+F"
    toggle_ball: str = "Ctrl+Shift+B"
    cycle_theme: str = "Ctrl+Shift+T"
    toggle_asr: str = "Ctrl+T"

    def to_dict(self) -> Dict[str, str]:
        return {
            "toggle_chat": self.toggle_chat,
            "region_screenshot": self.region_screenshot,
            "full_screenshot": self.full_screenshot,
            "toggle_ball": self.toggle_ball,
            "cycle_theme": self.cycle_theme,
            "toggle_asr": self.toggle_asr,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> "HotkeyConfig":
        return cls(
            toggle_chat=data.get("toggle_chat", "Ctrl+Shift+A"),
            region_screenshot=data.get("region_screenshot", "Ctrl+Shift+S"),
            full_screenshot=data.get("full_screenshot", "Ctrl+Shift+F"),
            toggle_ball=data.get("toggle_ball", "Ctrl+Shift+B"),
            cycle_theme=data.get("cycle_theme", "Ctrl+Shift+T"),
            toggle_asr=data.get("toggle_asr", "Ctrl+T"),
        )


class HotkeyManager(QObject):
    """快捷键管理器（支持应用级 + 系统级全局热键）。"""

    toggle_chat_triggered = Signal()
    region_screenshot_triggered = Signal()
    full_screenshot_triggered = Signal()
    toggle_ball_triggered = Signal()
    cycle_theme_triggered = Signal()
    asr_hotkey_pressed = Signal()
    asr_hotkey_released = Signal()
    action_dispatched = Signal(str)

    _instance: Optional["HotkeyManager"] = None
    _initialized: bool = False

    def __new__(cls, *args, **kwargs) -> "HotkeyManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        assert cls._instance is not None
        return cls._instance

    def __init__(self, parent: Optional[QWidget] = None):
        if self._initialized:
            return
        super().__init__(parent)
        self._initialized = True

        self._config = HotkeyConfig()
        self._shortcuts: Dict[str, QShortcut] = {}
        self._parent_widget = parent
        self._global_enabled = False

        self._global_hotkey_available = False
        self._keyboard = None
        self._keyboard_listener = None
        self._global_key_listener = None

        self._pressed_keys: set[int] = set()
        self._seq_cache: Dict[str, QKeySequence] = {}

        # 普通全局热键分发
        self.action_dispatched.connect(self._handle_dispatched_action)

        # ASR 全局热键状态
        self._asr_hotkey: str = "Ctrl+T"
        self._asr_hotkey_enabled: bool = True
        self._asr_required_keys: set[str] = set()
        self._asr_pressed_keys: set[str] = set()
        self._asr_recording_triggered: bool = False

        try:
            from pynput import keyboard  # type: ignore

            self._global_hotkey_available = True
            self._keyboard = keyboard
            logger.info("[HotkeyManager] 已启用系统级全局热键后端: pynput")
        except ImportError:
            logger.warning("[HotkeyManager] pynput 不可用，仅支持前台快捷键")

    def set_parent_widget(self, widget: QWidget):
        self._parent_widget = widget
        self._setup_qt_shortcuts()

    def set_config(self, config: HotkeyConfig):
        self._config = config
        self._asr_hotkey = (config.toggle_asr or "Ctrl+T").strip() or "Ctrl+T"
        logger.info("[HotkeyManager] 加载快捷键配置: %s", self._config.to_dict())
        self._setup_qt_shortcuts()
        if self._global_enabled:
            self._setup_global_hotkeys()
        self._setup_global_asr_hotkey()

    def set_asr_hotkey(self, hotkey: str, enabled: bool = True):
        self._asr_hotkey = (hotkey or "Ctrl+T").strip()
        self._asr_hotkey_enabled = bool(enabled)

        logger.info(
            "[HotkeyManager] 设置 ASR 热键: hotkey=%s enabled=%s",
            self._asr_hotkey,
            self._asr_hotkey_enabled,
        )

        self._setup_global_asr_hotkey()

    @property
    def is_asr_hotkey_global_active(self) -> bool:
        return bool(self._global_key_listener is not None)

    def get_config(self) -> HotkeyConfig:
        return self._config

    def _setup_qt_shortcuts(self):
        if not self._parent_widget:
            return

        for shortcut in self._shortcuts.values():
            shortcut.deleteLater()
        self._shortcuts.clear()

        shortcuts_map = {
            "toggle_chat": (self._config.toggle_chat, self.toggle_chat_triggered),
            "region_screenshot": (self._config.region_screenshot, self.region_screenshot_triggered),
            "full_screenshot": (self._config.full_screenshot, self.full_screenshot_triggered),
            "toggle_ball": (self._config.toggle_ball, self.toggle_ball_triggered),
            "cycle_theme": (self._config.cycle_theme, self.cycle_theme_triggered),
        }
        for name, (key_seq, signal) in shortcuts_map.items():
            if key_seq:
                shortcut = QShortcut(QKeySequence(key_seq), self._parent_widget)
                shortcut.activated.connect(signal.emit)
                self._shortcuts[name] = shortcut
        logger.info("[HotkeyManager] Qt 应用级快捷键已注册: %s", {k: v[0] for k, v in shortcuts_map.items()})

    def enable_global_hotkeys(self, enabled: bool = True):
        self._global_enabled = bool(enabled)
        logger.info(
            "[HotkeyManager] 全局热键开关: enabled=%s, backend_available=%s",
            self._global_enabled,
            self._global_hotkey_available,
        )

        if self._global_enabled and self._global_hotkey_available:
            logger.info("[HotkeyManager] 即将注册系统级全局热键")
            self._setup_global_hotkeys()
        else:
            logger.info("[HotkeyManager] 不注册系统级全局热键，转为停止监听")
            self._stop_global_hotkeys()

        # ASR 热键独立维护
        self._setup_global_asr_hotkey()



    def _setup_global_hotkeys(self):
        if not self._global_hotkey_available:
            logger.warning("[HotkeyManager] 全局热键不可用：pynput 未安装")
            return
        self._stop_global_hotkeys()

        action_shortcuts: Dict[str, str] = {
            "toggle_chat": self._config.toggle_chat,
            "region_screenshot": self._config.region_screenshot,
            "full_screenshot": self._config.full_screenshot,
            "toggle_ball": self._config.toggle_ball,
            "cycle_theme": self._config.cycle_theme,
        }
        signal_map = {
            "toggle_chat": self.toggle_chat_triggered,
            "region_screenshot": self.region_screenshot_triggered,
            "full_screenshot": self.full_screenshot_triggered,
            "toggle_ball": self.toggle_ball_triggered,
            "cycle_theme": self.cycle_theme_triggered,
        }

        mapping = {}
        for action, key_seq in action_shortcuts.items():
            combo = self._to_pynput_combo(key_seq)
            logger.info("[HotkeyManager] 注册热键: action=%s raw=%s combo=%s", action, key_seq, combo)
            if not combo:
                logger.warning("[HotkeyManager] 跳过无效快捷键: action=%s key=%s", action, key_seq)
                continue
            mapping[combo] = self._build_action_callback(signal_map[action], action)

        if not mapping:
            logger.warning("[HotkeyManager] 没有可注册的全局快捷键")
            return

        try:
            self._keyboard_listener = self._keyboard.GlobalHotKeys(mapping)
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
            logger.info("[HotkeyManager] 全局热键注册完成: %s", mapping)

        except Exception as e:
            logger.warning("[HotkeyManager] 全局热键注册失败: %s", e)
            self._keyboard_listener = None

    def _build_action_callback(self, signal, action: str):
        def _callback():
            logger.info("[HotkeyManager] pynput 回调命中: %s", action)
            self.action_dispatched.emit(action)

        return _callback

    def _handle_dispatched_action(self, action: str):
        logger.info("[HotkeyManager] 主线程处理热键动作: %s", action)

        signal_map = {
            "toggle_chat": self.toggle_chat_triggered,
            "region_screenshot": self.region_screenshot_triggered,
            "full_screenshot": self.full_screenshot_triggered,
            "toggle_ball": self.toggle_ball_triggered,
            "cycle_theme": self.cycle_theme_triggered,
        }

        signal = signal_map.get(action)
        if signal is None:
            logger.warning("[HotkeyManager] 未知热键动作: %s", action)
            return

        signal.emit()

    def _stop_global_hotkeys(self):
        if self._keyboard_listener:
            try:
                self._keyboard_listener.stop()
            except Exception as e:
                logger.debug("[HotkeyManager] 停止全局热键监听失败: %s", e)
            self._keyboard_listener = None

    def _to_pynput_combo(self, combo: str) -> Optional[str]:
        key_combo = (combo or "").strip().lower().replace(" ", "")
        if not key_combo:
            return None

        parts = [p for p in key_combo.split("+") if p]
        converted = []
        for part in parts:
            token = self._key_name_to_pynput_part(part)
            if not token:
                return None
            converted.append(token)

        return "+".join(converted) if converted else None

    def _key_name_to_pynput_part(self, key_name: str) -> Optional[str]:
        token = self._key_name_to_token(key_name)
        if not token:
            return None

        if token == "ctrl":
            return "<ctrl>"
        if token == "shift":
            return "<shift>"
        if token == "alt":
            return "<alt>"
        if token == "space":
            return "<space>"
        if token.startswith("f") and token[1:].isdigit():
            return f"<{token}>"

        return token

    def _key_name_to_token(self, key_name: str) -> Optional[str]:
        key_name = (key_name or "").strip().lower()

        if key_name in ("ctrl", "control"):
            return "ctrl"
        if key_name == "shift":
            return "shift"
        if key_name == "alt":
            return "alt"
        if key_name in ("space", "spacebar"):
            return "space"

        if len(key_name) == 1 and (key_name.isalpha() or key_name.isdigit()):
            return key_name.lower()

        if key_name.startswith("f") and key_name[1:].isdigit():
            n = int(key_name[1:])
            if 1 <= n <= 24:
                return f"f{n}"

        return None


    # def _emit_action(self, signal, action: str):
    #     logger.debug("[HotkeyManager] 收到热键事件: action=%s", action)
    #     signal.emit()

    def _setup_global_asr_hotkey(self):
        self._stop_global_asr_hotkey()

        if not self._asr_hotkey_enabled:
            logger.info("[HotkeyManager] ASR 全局快捷键已禁用")
            return

        if not self._global_hotkey_available:
            logger.warning("[HotkeyManager] 无法启用 ASR 全局快捷键：pynput 不可用")
            return

        required_keys = self._parse_key_tokens(self._asr_hotkey)
        if not required_keys:
            logger.warning("[HotkeyManager] 不支持的 ASR 快捷键: %s", self._asr_hotkey)
            return

        self._asr_required_keys = required_keys
        self._asr_pressed_keys = set()
        self._asr_combo_down = False

        logger.info(
            "[HotkeyManager] ASR 热键开始监听: hotkey=%s required=%s",
            self._asr_hotkey,
            sorted(self._asr_required_keys),
        )

        def on_press(key):
            key_token = self._normalize_pynput_key(key)
            if not key_token:
                return

            # 忽略自动重复
            if key_token in self._asr_pressed_keys:
                return

            before_pressed = set(self._asr_pressed_keys)
            before_combo_down = self._asr_combo_down

            self._asr_pressed_keys.add(key_token)

            logger.debug(
                "[HotkeyManager] ASR on_press: key=%s before=%s after=%s required=%s combo_down=%s",
                key_token,
                sorted(before_pressed),
                sorted(self._asr_pressed_keys),
                sorted(self._asr_required_keys),
                before_combo_down,
            )

            combo_now = self._asr_required_keys.issubset(self._asr_pressed_keys)

            # 只在第一次完整按下时触发
            if combo_now and not self._asr_combo_down:
                self._asr_combo_down = True
                logger.info("[HotkeyManager] ASR 按下命中，开始录音")
                self.asr_hotkey_pressed.emit()

        def on_release(key):
            key_token = self._normalize_pynput_key(key)
            if not key_token:
                return

            # 没记录过的键直接忽略
            if key_token not in self._asr_pressed_keys:
                return

            before_pressed = set(self._asr_pressed_keys)
            before_combo_down = self._asr_combo_down

            self._asr_pressed_keys.discard(key_token)

            logger.debug(
                "[HotkeyManager] ASR on_release: key=%s before=%s after=%s required=%s combo_down=%s",
                key_token,
                sorted(before_pressed),
                sorted(self._asr_pressed_keys),
                sorted(self._asr_required_keys),
                before_combo_down,
            )

            combo_now = self._asr_required_keys.issubset(self._asr_pressed_keys)

            # 只在组合键第一次被破坏时触发结束
            if self._asr_combo_down and not combo_now:
                self._asr_combo_down = False
                logger.info("[HotkeyManager] ASR 松开命中，结束录音")
                self.asr_hotkey_released.emit()

        try:
            self._global_key_listener = self._keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
            )
            self._global_key_listener.daemon = True
            self._global_key_listener.start()
            logger.info("[HotkeyManager] ASR 全局快捷键监听已启动: key=%s", self._asr_hotkey)
        except Exception as e:
            logger.warning("[HotkeyManager] 启动 ASR 全局快捷键失败: %s", e)
            self._global_key_listener = None

    def _parse_key_tokens(self, combo: str) -> set[str]:
        key_combo = (combo or "").strip().lower().replace(" ", "")
        if not key_combo:
            return set()

        result: set[str] = set()
        for part in [p for p in key_combo.split("+") if p]:
            token = self._key_name_to_token(part)
            if not token:
                return set()
            result.add(token)
        return result

    def _normalize_pynput_key(self, key) -> Optional[str]:
        if not self._keyboard:
            return None

        key_enum = getattr(self._keyboard, "Key", None)
        key_code_cls = getattr(self._keyboard, "KeyCode", None)

        if key_enum is not None:
            if key in (key_enum.ctrl, getattr(key_enum, "ctrl_l", None), getattr(key_enum, "ctrl_r", None)):
                return "ctrl"
            if key in (key_enum.shift, getattr(key_enum, "shift_l", None), getattr(key_enum, "shift_r", None)):
                return "shift"
            if key in (key_enum.alt, getattr(key_enum, "alt_l", None), getattr(key_enum, "alt_r", None),
                       getattr(key_enum, "alt_gr", None)):
                return "alt"
            if key == key_enum.space:
                return "space"

            key_name = getattr(key, "name", None)
            if isinstance(key_name, str) and key_name.startswith("f") and key_name[1:].isdigit():
                n = int(key_name[1:])
                if 1 <= n <= 24:
                    return key_name.lower()

        if key_code_cls is not None and isinstance(key, key_code_cls):
            vk = getattr(key, "vk", None)
            char = getattr(key, "char", None)

            # 先优先用可见字符
            if isinstance(char, str) and len(char) == 1 and char.isprintable() and char.isalnum():
                return char.lower()

            # Ctrl+字母时，char 可能是控制字符，例如 Ctrl+T -> \x14
            # 这种情况优先回退到 vk
            if isinstance(vk, int):
                if 65 <= vk <= 90:  # A-Z
                    return chr(vk).lower()
                if 48 <= vk <= 57:  # 0-9
                    return chr(vk)

            # 最后再兜底处理普通可打印字符
            if isinstance(char, str) and len(char) == 1 and char.isprintable():
                return char.lower()

        return None

    def _stop_global_asr_hotkey(self):
        if self._global_key_listener:
            logger.info("[HotkeyManager] 注销旧 ASR 全局快捷键监听器")
            try:
                self._global_key_listener.stop()
            except Exception as e:
                logger.debug("[HotkeyManager] 停止 ASR 全局热键监听失败: %s", e)
            self._global_key_listener = None

        self._asr_pressed_keys = set()
        self._asr_recording_triggered = False


    def cleanup(self):
        logger.info("[HotkeyManager] 清理热键资源")
        self._stop_global_hotkeys()
        self._stop_global_asr_hotkey()
        for shortcut in self._shortcuts.values():
            shortcut.deleteLater()
        self._shortcuts.clear()


_hotkey_manager: Optional[HotkeyManager] = None


def get_hotkey_manager() -> HotkeyManager:
    global _hotkey_manager
    if _hotkey_manager is None:
        _hotkey_manager = HotkeyManager()
    return _hotkey_manager


class _HotkeyManagerProxy:
    def __getattr__(self, name):
        return getattr(get_hotkey_manager(), name)


hotkey_manager = _HotkeyManagerProxy()  # type: ignore
