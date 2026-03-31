from __future__ import annotations

from pathlib import Path
from typing import Mapping


def generate_gpt_sovits_bat(output_dir: Path, cfg: Mapping[str, object]) -> Path:
    """生成 GPT-SoVITS 本地托管推理脚本（Windows .bat）。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    bat_path = output_dir / "start_gpt_sovits_local.bat"

    python_path = str(cfg.get("python_path", "python") or "python")
    api_script = str(cfg.get("api_script_path", "") or "")
    working_dir = str(cfg.get("working_dir", "") or "")
    tts_cfg = str(cfg.get("tts_config_path", "") or "")
    port = int(cfg.get("port", 9880) or 9880)

    script = f"""@echo off
setlocal enabledelayedexpansion

set "PYTHON_EXE={python_path}"
set "API_SCRIPT={api_script}"
set "WORK_DIR={working_dir}"
set "TTS_YAML={tts_cfg}"
set "PORT={port}"
set "HOST=127.0.0.1"

echo [INFO] GPT-SoVITS 本地托管启动脚本

where "%PYTHON_EXE%" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python 不可用: %PYTHON_EXE%
  exit /b 1
)

if not exist "%API_SCRIPT%" (
  echo [ERROR] API 脚本不存在: %API_SCRIPT%
  exit /b 1
)

if not exist "%WORK_DIR%" (
  echo [ERROR] 工作目录不存在: %WORK_DIR%
  exit /b 1
)

if not exist "%TTS_YAML%" (
  echo [ERROR] tts_infer.yaml 不存在: %TTS_YAML%
  exit /b 1
)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if not errorlevel 1 (
  echo [ERROR] 端口 %PORT% 已被占用，请更换端口
  exit /b 1
)

cd /d "%WORK_DIR%"
set "CMD=%PYTHON_EXE% %API_SCRIPT% -a %HOST% -p %PORT% -c %TTS_YAML%"
echo [INFO] 启动命令: !CMD!

call !CMD!
if errorlevel 1 (
  echo [ERROR] GPT-SoVITS 启动失败
  exit /b 1
)

echo [INFO] GPT-SoVITS 已退出
exit /b 0
"""

    bat_path.write_text(script, encoding="utf-8", newline="\r\n")
    return bat_path
