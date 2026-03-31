# GPT-SoVITS 本地托管推理（api_v2.py / localhost HTTP）

> 本文基于 `RVC-Boss/GPT-SoVITS` 官方仓库当前 `main` 分支的 `api_v2.py` 与 README。

## 接入边界

- 仅接入本地托管推理端（`api_v2.py`）。
- 仅绑定 `127.0.0.1`。
- 客户端仅通过 localhost HTTP 调用。
- 不接 `webui.py`，不依赖 Gradio。
- 不接训练端/数据处理端/训练 WebUI。

## 官方 `api_v2.py` 关键点（对齐结果）

1. 启动参数：
   - `-c / --tts_config`
   - `-a / --bind_addr`
   - `-p / --port`
2. 主要接口：
   - `POST /tts`（也有 `GET /tts`）
   - `GET /set_gpt_weights?weights_path=...`
   - `GET /set_sovits_weights?weights_path=...`
   - `GET /control?command=...`
3. `/health`：官方代码中不保证存在，不能写死依赖。

## 客户端健康检查策略（自动探测）

按顺序探测（任一成功即认为服务可用）：
1. 用户配置 health endpoint（若有）
2. `/health`
3. `/docs`
4. `/openapi.json`
5. `/control`（仅连通性探测，不执行命令）

## 客户端请求字段（对齐 `POST /tts`）

客户端默认发送：
- `text`
- `text_lang`
- `ref_audio_path`
- `prompt_lang`
- `prompt_text`
- `text_split_method`
- `media_type=wav`
- `streaming_mode=false`

并将 roaming/config 中的额外参数（JSON）合并到请求体。

## 设置页只保留的 GPT-SoVITS 客户端字段

- GPT 权重选择（从配置目录自动扫描）
- SoVITS 权重选择（从配置目录自动扫描）
- 参考语音
- 参考语音文字

> 其他运行参数（python/api脚本/工作目录/端口/超时/切句扩展参数等）全部走 roaming/config，不在 UI 暴露。

## Windows 本地运行建议（官方方向）

1. 准备 Python 环境与依赖。
2. 准备 `api_v2.py`、权重与 `tts_infer.yaml`。
3. 使用：
   - `python api_v2.py -a 127.0.0.1 -p 9880 -c GPT_SoVITS/configs/tts_infer.yaml`
4. 客户端再通过 `http://127.0.0.1:9880` 调用。

## 常见问题

### Q1: 为什么不直接用 `/health`？
A: 官方 `api_v2.py` 版本中不保证存在 `/health`，因此实现了自动探测。

### Q2: 权重切换如何做？
A: 启动后调用 `/set_gpt_weights` 与 `/set_sovits_weights`，参数 `weights_path` 为绝对路径。

### Q3: 客户端是否统一切句？
A: 不做客户端统一切句。GPT-SoVITS 侧通过 `text_split_method` 与扩展参数自行处理。
