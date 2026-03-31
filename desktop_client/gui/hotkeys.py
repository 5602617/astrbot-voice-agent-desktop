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
    quick_ask: str = "Ctrl+Shift+Q"
    cycle_theme: str = "Ctrl+Shift+T"
    toggle_asr: str = "Ctrl+T"

    def to_dict(self) -> Dict[str, str]:
        return {
            "toggle_chat": self.toggle_chat,
            "region_screenshot": self.region_screenshot,
            "full_screenshot": self.full_screenshot,
            "toggle_ball": self.toggle_ball,
            "quick_ask": self.quick_ask,
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
            quick_ask=data.get("quick_ask", "Ctrl+Shift+Q"),
            cycle_theme=data.get("cycle_theme", "Ctrl+Shift+T"),
            toggle_asr=data.get("toggle_asr", "Ctrl+T"),
        )


class HotkeyManager(QObject):
    """快捷键管理器（支持应用级 + 系统级全局热键）。"""

    toggle_chat_triggered = Signal()
    region_screenshot_triggered = Signal()
    full_screenshot_triggered = Signal()
    toggle_ball_triggered = Signal()
    quick_ask_triggered = Signal()
    cycle_theme_triggered = Signal()
    asr_hotkey_pressed = Signal()
    asr_hotkey_released = Signal()

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

        self._asr_hotkey: str = "Ctrl+T"
        self._asr_hotkey_enabled: bool = True
        self._asr_hotkey_vk: Optional[int] = None

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

    def set_asr_hotkey(self, key: str, enabled: bool = True):
        self._asr_hotkey = (key or "Ctrl+T").strip() or "Ctrl+T"
        self._asr_hotkey_enabled = bool(enabled)
        logger.info(
            "[HotkeyManager] 配置录音快捷键: key=%s enabled=%s",
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
            "quick_ask": (self._config.quick_ask, self.quick_ask_triggered),
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
        logger.info("[HotkeyManager] 全局热键开关: enabled=%s", self._global_enabled)

        if self._global_enabled and self._global_hotkey_available:
            self._setup_global_hotkeys()
        else:
            self._stop_global_hotkeys()

        self._setup_global_asr_hotkey()

    def _setup_global_hotkeys(self):
        if not self._global_hotkey_available:
            logger.warning("[HotkeyManager] 全局热键不可用：pynput 未安装")
            return
        self._stop_global_hotkeys()

        action_map: dict[int, Callable[[], None]] = {}
        action_shortcuts: Dict[str, str] = {
            "toggle_chat": self._config.toggle_chat,
            "region_screenshot": self._config.region_screenshot,
            "full_screenshot": self._config.full_screenshot,
            "toggle_ball": self._config.toggle_ball,
            "quick_ask": self._config.quick_ask,
            "cycle_theme": self._config.cycle_theme,
        }
        signal_map = {
            "toggle_chat": self.toggle_chat_triggered,
            "region_screenshot": self.region_screenshot_triggered,
            "full_screenshot": self.full_screenshot_triggered,
            "toggle_ball": self.toggle_ball_triggered,
            "quick_ask": self.quick_ask_triggered,
            "cycle_theme": self.cycle_theme_triggered,
        }

        parsed: dict[str, tuple[bool, bool, bool, int]] = {}
        for action, key_seq in action_shortcuts.items():
            combo = self._parse_combo(key_seq)
            if combo is None:
                logger.warning("[HotkeyManager] 跳过无效快捷键: action=%s key=%s", action, key_seq)
                continue
            need_ctrl, need_shift, need_alt, main_vk = combo
            parsed[action] = (need_ctrl, need_shift, need_alt, main_vk)
            action_map[main_vk] = lambda sig=signal_map[action], a=action: self._emit_action(sig, a)

        def on_press(key):
            vk = getattr(key, "vk", None)
            if vk is None:
                return
            self._pressed_keys.add(vk)
            for action, (need_ctrl, need_shift, need_alt, main_vk) in parsed.items():
                if vk == main_vk and self._mods_match(need_ctrl, need_shift, need_alt):
                    self._emit_action(signal_map[action], action)

        def on_release(key):
            vk = getattr(key, "vk", None)
            if vk is not None and vk in self._pressed_keys:
                self._pressed_keys.discard(vk)

        try:
            self._keyboard_listener = self._keyboard.Listener(on_press=on_press, on_release=on_release)
            self._keyboard_listener.daemon = True
            self._keyboard_listener.start()
            logger.info("[HotkeyManager] 全局热键注册完成: %s", action_shortcuts)
            logger.info("[HotkeyManager] action -> shortcut 映射: %s", action_shortcuts)
        except Exception as e:
            logger.warning("[HotkeyManager] 全局热键注册失败: %s", e)
            self._keyboard_listener = None

    def _emit_action(self, signal, action: str):
        logger.debug("[HotkeyManager] 收到热键事件: action=%s", action)
        signal.emit()

    def _stop_global_hotkeys(self):
        if self._keyboard_listener:
            logger.info("[HotkeyManager] 注销旧全局热键监听器")
            try:
                self._keyboard_listener.stop()
            except Exception:
                pass
            self._keyboard_listener = None
        self._pressed_keys.clear()

    def _setup_global_asr_hotkey(self):
        self._stop_global_asr_hotkey()
        if not self._asr_hotkey_enabled:
            logger.info("[HotkeyManager] ASR 全局快捷键已禁用")
            return
        if not self._global_hotkey_available:
            logger.warning("[HotkeyManager] 无法启用 ASR 全局快捷键：pynput 不可用")
            return

        combo = self._parse_combo(self._asr_hotkey)
        if combo is None:
            logger.warning("[HotkeyManager] 不支持的 ASR 快捷键: %s", self._asr_hotkey)
            return
        need_ctrl, need_shift, need_alt, main_vk = combo
        self._asr_hotkey_vk = main_vk
        pressed = {"recording_triggered": False}

        def on_press(key):
            vk = getattr(key, "vk", None)
            if vk is None:
                return
            self._pressed_keys.add(vk)
            if vk == main_vk and self._mods_match(need_ctrl, need_shift, need_alt):
                if not pressed["recording_triggered"]:
                    pressed["recording_triggered"] = True
                    logger.debug("[HotkeyManager] 收到 ASR 按下事件")
                    self.asr_hotkey_pressed.emit()

        def on_release(key):
            vk = getattr(key, "vk", None)
            if vk is not None:
                self._pressed_keys.discard(vk)
            if vk == main_vk and pressed["recording_triggered"]:
                pressed["recording_triggered"] = False
                logger.debug("[HotkeyManager] 收到 ASR 松开事件")
                self.asr_hotkey_released.emit()

        try:
            self._global_key_listener = self._keyboard.Listener(on_press=on_press, on_release=on_release)
            self._global_key_listener.daemon = True
            self._global_key_listener.start()
            logger.info("[HotkeyManager] ASR 全局快捷键监听已启动: key=%s", self._asr_hotkey)
        except Exception as e:
            logger.warning("[HotkeyManager] 启动 ASR 全局快捷键失败: %s", e)
            self._global_key_listener = None

    def _stop_global_asr_hotkey(self):
        if self._global_key_listener:
            logger.info("[HotkeyManager] 注销旧 ASR 全局快捷键监听器")
            try:
                self._global_key_listener.stop()
            except Exception:
                pass
            self._global_key_listener = None

    def _parse_combo(self, combo: str) -> Optional[tuple[bool, bool, bool, int]]:
        key_combo = (combo or "").strip().lower().replace(" ", "")
        if not key_combo:
            return None
        parts = [p for p in key_combo.split("+") if p]
        need_ctrl = False
        need_shift = False
        need_alt = False
        main_vk: Optional[int] = None
        for p in parts:
            if p in ("ctrl", "control"):
                need_ctrl = True
            elif p == "shift":
                need_shift = True
            elif p == "alt":
                need_alt = True
            else:
                main_vk = self._resolve_vk_from_key(p)
        if main_vk is None:
            return None
        return need_ctrl, need_shift, need_alt, main_vk

    def _resolve_vk_from_key(self, key_name: str) -> Optional[int]:
        if key_name == "space":
            return 32
        if len(key_name) == 1 and key_name.isalpha():
            return ord(key_name.upper())
        if len(key_name) == 1 and key_name.isdigit():
            return ord(key_name)
        if key_name.startswith("f") and key_name[1:].isdigit():
            n = int(key_name[1:])
            if 1 <= n <= 24:
                return 111 + n
        return None

    def _mods_match(self, need_ctrl: bool, need_shift: bool, need_alt: bool) -> bool:
        has_ctrl = (162 in self._pressed_keys) or (163 in self._pressed_keys)
        has_shift = (160 in self._pressed_keys) or (161 in self._pressed_keys)
        has_alt = (164 in self._pressed_keys) or (165 in self._pressed_keys)
        return has_ctrl == need_ctrl and has_shift == need_shift and has_alt == need_alt

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
