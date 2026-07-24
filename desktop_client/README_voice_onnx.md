# ONNX 模型目录与 GenieData 配置说明（桌面端）

本文用于说明 AstrBot 桌面客户端中，本地 Genie-TTS 相关目录如何配置，重点区分：

- **GenieData 公共资源目录**（`GENIE_DATA_DIR`）
- **角色 ONNX 模型目录**（`onnx_model_dir`）

---

## 1. GenieData 是什么？

`GenieData` 是 `genie-tts` 运行时使用的**公共资源目录**，不是某个角色专属模型目录。

通常会包含（不同版本文件名可能略有差异）：

- `G2P/`
- `chinese-hubert-base/`
- `speaker_encoder.onnx`

> 你在设置页中填写的 `GENIE_DATA_DIR`，应当指向这个公共目录。

示例：

- `D:\Astrbot-desktop-assistant\GenieData`

---

## 2. 预置角色模式如何工作？

在“预置角色模式”下（例如 `feibi`）：

- 客户端会调用类似 `load_predefined_character("feibi")`
- 通常**不需要**用户手动填写角色 ONNX 目录
- 只要 `GenieData` 可用，预置角色即可工作

适合普通用户的推荐配置：

1. 选择“预置角色模式”
2. 角色名填写 `feibi`（或你使用的预置名）
3. 正确配置 `GENIE_DATA_DIR`
4. 不填写 ONNX 模型目录也可以

---

## 3. 自定义角色模式如何工作？

在“本地 ONNX 模型模式”下：

- 你需要先将训练产物（例如 `.pth` + `.ckpt`）转换得到 ONNX 导出目录（常见名 `onnx_out`）
- 在设置页里填写该目录到“ONNX 模型目录”

示例：

- `D:\models\myvoice\onnx_out`

---

## 4. 角色 ONNX 目录结构示例（建议）

结合当前客户端常见校验与社区导出目录，建议基础文件至少包含：

```text
onnx_out/
├─ t2s_encoder_fp32.bin
├─ t2s_encoder_fp32.onnx
├─ t2s_first_stage_decoder_fp32.onnx
├─ t2s_shared_fp16.bin
├─ t2s_stage_decoder_fp32.onnx
├─ vits_fp16.bin
└─ vits_fp32.onnx
```

若是 `v2ProPlus` 目录，通常还需要：

```text
onnx_out/
├─ prompt_encoder_fp16.bin
└─ prompt_encoder_fp32.onnx
```

> 实际文件仍以你使用的导出脚本版本为准；客户端会把目录传给 `genie-tts` 运行时加载。

---

## 5. GENIE_DATA_DIR 与 onnx_model_dir 的区别

- `GENIE_DATA_DIR`：
  - 指向 **GenieData 公共资源目录**
  - 通常全角色共用

- `onnx_model_dir`：
  - 指向 **某个自定义角色的 ONNX 导出目录**
  - 仅在“本地 ONNX 模型模式”下需要

两者不是同一个目录，不能互相替代。

---

## 6. 设置页字段填写建议

- **Genie 模式**
  - 普通用户：`预置角色模式`
  - 高级用户：`本地 ONNX 模型模式`

- **预置角色名**
  - 预置模式下填写，例如 `feibi`

- **角色名**
  - ONNX 模式下填写自定义角色标识

- **ONNX 模型目录**
  - 仅 ONNX 模式下填写，例如 `D:\models\myvoice\onnx_out`

- **GENIE_DATA_DIR(可选)**
  - 填 GenieData 目录，例如 `<project_root>\GenieData`

---

## 7. 常见错误

1. 把 `GENIE_DATA_DIR` 指到角色模型目录
2. 把角色模型目录指到 `GenieData`
3. 预置模式误以为必须填写 ONNX 模型目录

排查思路：先确认 `GENIE_DATA_DIR` 正确，再确认当前模式和对应字段是否匹配。
