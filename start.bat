@echo off
chcp 65001 >nul
title AstrBot 桌面助手

set "PYTHON_EXE=D:\conda\envs\dasktop\python.exe"

echo ============================================
echo    AstrBot 桌面助手启动器
echo ============================================
echo.

:: 切换到脚本所在目录
cd /d "%~dp0"

:: 检查指定 Python 是否存在
echo [1/4] 检查 Python...
if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到指定 Python:
    echo %PYTHON_EXE%
    pause
    exit /b 1
)

"%PYTHON_EXE%" --version
if errorlevel 1 (
    echo [错误] Python 无法正常运行
    pause
    exit /b 1
)

:: 检查依赖
echo [2/4] 检查依赖...
"%PYTHON_EXE%" -m pip show PySide6 >nul 2>&1
if errorlevel 1 (
    echo [3/4] 安装依赖中，请稍候...
    "%PYTHON_EXE%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [错误] 依赖安装失败，请检查网络连接或 requirements.txt
        pause
        exit /b 1
    )
    echo [3/4] 依赖安装完成!
) else (
    echo [3/4] 依赖已就绪
)

echo [4/4] 启动应用...
echo.

:: 启动应用
"%PYTHON_EXE%" -m desktop_client

:: 如果程序异常退出，显示错误信息
if errorlevel 1 (
    echo.
    echo [错误] 应用异常退出，错误代码: %errorlevel%
    pause
)