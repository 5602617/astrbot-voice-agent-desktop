#!/usr/bin/env python3
"""SoVITS runtime wrapper (legacy compatibility path).

最小参数协议：
--text --output --model-dir --ref-audio --prompt-text --prompt-lang --text-lang [--speaker]

说明：
- 若本地存在可用 GPT-SoVITS Python API，可在 `_try_gpt_sovits` 中扩展接入。
- 默认 fallback 到 pyttsx3 生成 wav，确保链路可跑。
- 注意：当前主链路已迁移到内部 runtime provider，请仅在 legacy mode 下使用本脚本。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _try_gpt_sovits(args) -> bool:
    """尝试接入本地 GPT-SoVITS Python API（按你的实际环境补充）。"""
    # 这里保留明确扩展点，避免硬编码某一份第三方仓库结构
    return False


def _fallback_pyttsx3(text: str, output: Path) -> None:
    try:
        import pyttsx3
    except Exception as exc:
        raise RuntimeError("缺少 pyttsx3，且未检测到可用 SoVITS runtime API") from exc

    engine = pyttsx3.init()
    engine.save_to_file(text, str(output))
    engine.runAndWait()


def main() -> int:
    parser = argparse.ArgumentParser(description="SoVITS runtime wrapper")
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--ref-audio", default="")
    parser.add_argument("--prompt-text", default="")
    parser.add_argument("--prompt-lang", default="")
    parser.add_argument("--text-lang", default="zh")
    parser.add_argument("--speaker", default="")

    args = parser.parse_args()
    out = Path(args.output)
    _ensure_parent(out)

    if _try_gpt_sovits(args):
        return 0

    _fallback_pyttsx3(args.text, out)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"wrapper 生成失败: {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"[sovits_wrapper] error: {e}", file=sys.stderr)
        raise
