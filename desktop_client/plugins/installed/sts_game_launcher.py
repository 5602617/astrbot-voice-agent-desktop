import asyncio
import logging
import os
import subprocess
from pathlib import Path

from desktop_client.plugins.base import IPlugin, PluginMetadata
from desktop_client.plugins.hooks import HookContext, HookPriority, HookResult, HookType

logger = logging.getLogger(__name__)

STEAM_URI = "steam://rungameid/646570"
MOD_LAUNCHER = r"E:\SteamLibrary\steamapps\common\SlayTheSpire\start_sts_mod_mode.bat"

LAUNCH_KEYWORDS = ("开", "打开", "启动", "运行")
GAME_KEYWORDS = ("杀戮尖塔", "slay", "sts")
MOD_KEYWORDS = ("mod", "agent", "模组")


class STSGameLauncherPlugin(IPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="sts_game_launcher",
            version="1.0.0",
            author="OpenAI",
            description="拦截聊天指令并启动杀戮尖塔普通模式或 Mod 模式",
            tags=["game", "launcher", "slay_the_spire"],
        )

    def on_load(self) -> bool:
        self.register_hook(
            HookType.PRE_MESSAGE_SEND,
            self._on_pre_message_send,
            HookPriority.HIGHEST,
        )
        logger.info("STSGameLauncherPlugin 已加载")
        return True

    async def _on_pre_message_send(self, context: HookContext) -> HookResult:
        text = (context.get("message", "") or "").strip()
        if not text:
            return HookResult.CONTINUE

        launch_hit, game_hit, mod_hit = self._analyze_text(text)

        logger.info(
            "STS 命中检测: text=%s launch_hit=%s game_hit=%s mod_hit=%s",
            text,
            launch_hit,
            game_hit,
            mod_hit,
        )

        if launch_hit and game_hit:
            if mod_hit:
                await self._launch_mod_mode()
            else:
                await self._launch_normal_mode()
            return HookResult.ABORT

        return HookResult.CONTINUE

    def _analyze_text(self, text: str) -> tuple[bool, bool, bool]:
        normalized = text.lower()

        launch_hit = any(keyword in normalized for keyword in LAUNCH_KEYWORDS)
        game_hit = any(keyword in normalized for keyword in GAME_KEYWORDS)
        mod_hit = any(keyword in normalized for keyword in MOD_KEYWORDS)

        return launch_hit, game_hit, mod_hit

    async def _launch_normal_mode(self) -> None:
        try:
            await asyncio.to_thread(os.startfile, STEAM_URI)
            logger.info("已启动普通模式: %s", STEAM_URI)
        except Exception:
            logger.exception("启动普通模式失败")

    async def _launch_mod_mode(self) -> None:
        try:
            path = Path(MOD_LAUNCHER)
            if not path.exists():
                raise FileNotFoundError(f"未找到 Mod 启动脚本: {MOD_LAUNCHER}")

            def _start_bat():
                subprocess.Popen(
                    ["cmd", "/c", "start", "", str(path)],
                    cwd=str(path.parent),
                    shell=False,
                )

            await asyncio.to_thread(_start_bat)
            logger.info("已启动 Mod 模式: %s", MOD_LAUNCHER)
        except Exception:
            logger.exception("启动 Mod 模式失败")