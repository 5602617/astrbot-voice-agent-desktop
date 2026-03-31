# GPT-SoVITS 本地托管推理（方案 A）

## 功能说明

桌面端可在本机启动 GPT-SoVITS **推理 API**，并通过 `http://127.0.0.1:<port>` 进行 TTS 合成调用。

> 本功能只支持推理端，不支持训练端、数据处理端和训练 WebUI。

## 范围限制（必须遵守）

- 仅本地托管推理端。
- 仅绑定 `127.0.0.1`。
- 客户端仅通过 localhost HTTP 调用。
- 不启用本地 HTTPS。
- 不接入训练相关模块。

## 启用前准备

1. 已可运行的 GPT-SoVITS 推理代码目录。
2. 可执行 Python（例如 `python` 或 `C:/Python311/python.exe`）。
3. `api_v2.py` 完整路径。
4. 工作目录（通常是 GPT-SoVITS 项目根目录）。
5. `tts_infer.yaml` 路径。
6. 未占用端口（默认 `9880`）。

## 设置项说明（GPT-SoVITS Provider）

- **GPT-SoVITS Python**：Python 可执行路径。
- **GPT-SoVITS API脚本**：`api_v2.py` 的完整路径。
- **GPT-SoVITS 工作目录**：启动子进程时的 cwd。
- **GPT-SoVITS 端口**：本地推理服务端口（通常 9880）。
- **启动超时**：等待服务健康检查通过的最大秒数。
- **健康检查超时**：单次健康检查请求超时。
- **请求超时**：TTS 合成请求超时。
- **tts_infer.yaml 路径**：推理配置文件路径。
- **默认语言**：例如 `zh` / `en` / `ja`。
- **参考音频 / 参考文本**：需要参考音色时填写，可选。
- **切句模式**：传给 GPT-SoVITS 的分句策略（如 `auto`）。
- **切句参数(JSON)**：额外切句参数，例如 `{"max_len":120}`。
- **自动关闭子进程**：客户端退出时是否自动结束本地推理进程。

## API脚本、工作目录、tts_infer.yaml 应该填什么

- API脚本：`.../GPT-SoVITS/api_v2.py`
- 工作目录：`.../GPT-SoVITS`
- tts_infer.yaml：`.../GPT-SoVITS/GPT_SoVITS/configs/tts_infer.yaml`

## 初始化脚本生成

设置页提供 **“生成 GPT-SoVITS 初始化脚本”** 按钮。

生成路径：
- `<配置目录>/scripts/gpt_sovits/start_gpt_sovits_local.bat`

该脚本会检查：
- Python 是否可用
- `api_v2.py` 是否存在
- 工作目录是否存在
- `tts_infer.yaml` 是否存在
- 端口是否被占用

然后自动拼接并执行：
- `python api_v2.py --host 127.0.0.1 --port <port> --config <tts_infer.yaml>`

## 启动失败排查

1. Python 不可用：检查 Python 路径是否正确。
2. API脚本不存在：确认 `api_v2.py` 路径。
3. 工作目录错误：确认目录存在且可访问。
4. tts_infer.yaml 不存在：确认配置路径。
5. 端口占用：更换端口后重试。

## 健康检查失败排查

1. 启动日志中是否有 Python 异常。
2. 端口是否真的监听在 `127.0.0.1`。
3. 健康检查 endpoint 是否与服务实现一致（默认 `/health`）。
4. 将启动超时调大后重试。

## 如何验证推理端启动成功

- 浏览器或 curl 访问：`http://127.0.0.1:<port>/health`
- 返回 2xx/4xx（非 5xx）且客户端日志显示健康检查通过，即视为就绪。

## 如何切换 Provider

- 切换到 GPT-SoVITS：设置页 `TTS Provider = GPT-SoVITS(本地托管)`。
- 切回 Genie-TTS：设置页 `TTS Provider = Genie-TTS`。

切换后 UI 会立即隐藏不相关配置项，不会清除已保存值。

## FAQ

### Q1: 为什么不支持训练端？
A: 当前方案专注桌面端本地推理，避免引入训练链路和大体量依赖。

### Q2: 可以绑定 0.0.0.0 吗？
A: 不支持。当前仅允许 `127.0.0.1` 以降低暴露风险。

### Q3: 客户端会做统一切句吗？
A: 不会。客户端不再做统一切句，GPT-SoVITS 使用自身切句逻辑。
