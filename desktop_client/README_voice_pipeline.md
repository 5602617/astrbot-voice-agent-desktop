# Voice Pipeline 使用文档

## 1. 功能简介

本项目在桌面客户端内新增了可替换的语音流水线：

- 音频输入（文件/bytes）
- ASR 识别（HTTP 或 Runtime）
- 复用现有文本发送链路进入云端 LLM
- 监听 LLM 回复 chunk/end
- 回复结束触发 TTS（HTTP 或 Runtime）
- 音频缓存与可选自动播放

该语音链路不会替换原有文本链路，文本功能保持不变。

---

## 2. 当前语音链路说明

```text
audio(file/bytes)
  -> VoicePipelineRuntime.handle_audio_*
  -> ASR Provider
  -> handle_asr_text
  -> MessageBridge.send_input(text)
  -> 云端 AstrBot LLM
  -> SSE text chunk / end
  -> VoicePipelineRuntime.on_llm_reply_chunk/end
  -> TTS Provider
  -> audio file (cache)
```

---

## 3. 与现有文本链路如何集成

未改动核心链路：

- 发送：`_on_message_sent -> MessageBridge.send_input`
- 接收：`_handle_sse_event -> _on_bridge_message -> MessageHandler`

语音通过 `local_voice_bridge` 插件在 hook 层接入：

- `PRE_MESSAGE_SEND`：可中断上一轮语音
- `POST_MESSAGE_RECEIVE`：收集 LLM 回复并在 end 时触发 TTS

---

## 4. 新增目录结构

```text
desktop_client/services/voice_pipeline/
  __init__.py
  base.py
  models.py
  registry.py
  pipeline.py
  turn_manager.py
  providers/
    __init__.py
    noop.py
    http_asr.py
    http_tts.py
    runtime_asr.py
    runtime_tts.py
```

---

## 5. 配置项说明（VoiceConfig）

### 总开关
- `enable_voice_pipeline`
- `interrupt_tts_on_new_input`
- `interrupt_asr_on_new_input`
- `auto_play_tts`
- `emit_asr_text_message`
- `save_audio_cache`
- `audio_cache_dir`

### ASR
- `asr_enabled`
- `asr_provider_type` (`none/http/runtime`)
- `asr_api_url`
- `asr_timeout`
- `asr_upload_field`
- `asr_response_text_key`
- `asr_headers_json`
- `asr_extra_params_json`
- `asr_runtime_backend` (`faster_whisper/sherpa_onnx/funasr/custom`)
- `asr_model_path`
- `asr_tokens_path`
- `asr_device`
- `asr_language`

### TTS
- `tts_enabled`
- `tts_provider_type` (`none/http/runtime`)
- `tts_api_url`
- `tts_method` (`GET/POST`)
- `tts_text_field`
- `tts_response_mode` (`audio_stream/json_url/json_file/json_base64`)
- `tts_response_key`
- `tts_headers_json`
- `tts_extra_params_json`
- `tts_audio_format`
- `tts_runtime_backend` (`qt/pyttsx3/edge_tts/gpt_sovits/custom`)
- `tts_model_path`
- `tts_speaker`
- `tts_language`

---

## 6. 外部 HTTP ASR 接入示例

```json
{
  "asr_enabled": true,
  "asr_provider_type": "http",
  "asr_api_url": "http://127.0.0.1:12394/asr/wav",
  "asr_upload_field": "file",
  "asr_response_text_key": "text",
  "asr_headers_json": "{}",
  "asr_extra_params_json": "{}"
}
```

---

## 7. 外部 HTTP TTS 接入示例

```json
{
  "tts_enabled": true,
  "tts_provider_type": "http",
  "tts_api_url": "http://127.0.0.1:9880/tts",
  "tts_method": "POST",
  "tts_text_field": "text",
  "tts_response_mode": "audio_stream",
  "tts_audio_format": "wav",
  "tts_extra_params_json": "{\"text_lang\":\"zh\"}"
}
```

---

## 8. 内部 runtime 模式说明

### ASR Runtime
- 已完整实现：`faster_whisper`, `sherpa_onnx`（优先尝试 `backend.asr.sherpa_asr.SherpaASR`）
- 扩展骨架：`funasr`, `custom`

### TTS Runtime
- 已完整可运行：`qt`（实时说话）、`pyttsx3`（导出文件）、`edge_tts`（导出文件）
- SoVITS Runtime：`sovits` / `gpt_sovits`（wrapper 模式，可运行，需外部脚本）
- 扩展骨架：`custom`

---

## 9. 模型文件如何放置

- `asr_model_path`：可填写 faster-whisper 模型名（如 `base`）或本地目录
- `tts_model_path`：给自定义后端使用（当前 qt/pyttsx3/edge_tts 不强依赖）

建议把大型模型放在独立目录，再通过配置引用路径。

---

## 10. 如何启动和测试

1. 启动桌面客户端
2. 打开语音配置（config.json）启用 pipeline
3. 调用：
   - `DesktopClientApp.submit_local_asr_text(...)`
   - 或 `submit_local_asr_audio_file(...)`
   - 或 `submit_local_asr_audio_bytes(...)`
4. 观察日志：
   - provider 初始化
   - ASR 结果长度
   - TTS 输出路径

### 界面触发 ASR

- 聊天输入区新增 🎤 按钮（可由 `enable_asr_button` 控制）
- 快捷键新增 `toggle_asr`（默认 `Ctrl+Shift+R`，可由 `enable_asr_hotkey` 控制）
- 第一次触发开始录音，第二次触发停止并自动提交识别

### 配置 SherpaASR Runtime

```json
{
  "voice": {
    "enable_voice_pipeline": true,
    "asr_enabled": true,
    "asr_provider_type": "runtime",
    "asr_runtime_backend": "sherpa_onnx",
    "asr_model_path": "models/sherpa",
    "asr_tokens_path": "models/sherpa/tokens.txt"
  }
}
```

### 配置 SoVITS Runtime

```json
{
  "voice": {
    "tts_enabled": true,
    "tts_provider_type": "runtime",
    "tts_runtime_backend": "sovits",
    "tts_runtime_python": "python",
    "tts_runtime_script": "scripts/sovits_wrapper.py",
    "tts_model_path": "models/sovits",
    "tts_ref_audio_path": "assets/ref.wav",
    "tts_prompt_text": "示例提示词",
    "tts_prompt_lang": "zh"
  }
}
```

---

## 11. 常见问题排查

1. `faster_whisper` 导入失败
   - 安装依赖：`pip install faster-whisper`

2. `sherpa_onnx` 初始化失败
   - 检查 `backend.asr.sherpa_asr.SherpaASR` 是否可导入
   - 检查 `asr_model_path` / `asr_tokens_path` 路径
   - 若缺依赖：`pip install sherpa-onnx`

3. HTTP ASR 返回空文本
   - 检查 `asr_response_text_key`
   - 检查服务端返回 JSON 结构

4. HTTP TTS 没有音频
   - 检查 `tts_response_mode` 是否匹配服务端
   - `audio_stream`/`json_url`/`json_file`/`json_base64`

5. SoVITS runtime 失败
   - 检查 `tts_runtime_script` 路径
   - 手动执行脚本确认参数 `--text --output` 可用

6. 语音失败影响文本
   - 正常不应发生；provider 会回退 noop，不影响文本主链路

7. 配置写了 runtime 但仍走 http
   - 检查 `asr_provider_type` / `tts_provider_type` 是否为 `runtime`
   - 观察启动日志中的 `asr_cls/tts_cls`

---

## 12. 如何扩展自定义 ASR provider

1. 继承 `BaseASRProvider`
2. 实现 `transcribe_file` / `transcribe_bytes`
3. 在 `registry.py` 增加构造分支
4. 在 `VoiceConfig` 中增加 provider_type/backend 配置

---

## 13. 如何扩展自定义 TTS provider

1. 继承 `BaseTTSProvider`
2. 实现 `synthesize_to_file` 或 `synthesize_bytes`
3. 在 `registry.py` 增加构造分支
4. 在配置中选择新的 provider_type/backend

---

## 兼容性说明

- `LocalVoiceRuntime` 仍保留，内部委托给 `VoicePipelineRuntime`
- `local_voice_bridge.py` 保留且改为薄桥接层
- 现有文本、截图、媒体、UI 渲染流程保持原有行为
