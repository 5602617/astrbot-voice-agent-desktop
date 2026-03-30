from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class ASRProviderConfig:
    enabled: bool = False
    provider_type: str = "none"  # none / http / runtime
    api_url: str = ""
    timeout: int = 30
    upload_field: str = "file"
    response_text_key: str = "text"
    headers_json: str = "{}"
    extra_params_json: str = "{}"
    runtime_backend: str = "faster_whisper"  # sherpa_onnx / faster_whisper / funasr / custom
    model_path: str = ""
    tokens_path: str = ""
    device: str = "cpu"
    language: str = "zh"


@dataclass
class TTSProviderConfig:
    enabled: bool = False
    provider_type: str = "none"  # none / http / runtime
    api_url: str = ""
    method: str = "POST"  # GET / POST
    text_field: str = "text"
    response_mode: str = "audio_stream"  # audio_stream / json_url / json_file / json_base64
    response_key: str = "audio_url"
    headers_json: str = "{}"
    extra_params_json: str = "{}"
    audio_format: str = "wav"
    runtime_backend: str = "qt"  # qt / pyttsx3 / edge_tts / gpt_sovits / custom
    model_path: str = ""
    speaker: str = ""
    language: str = "zh"


@dataclass
class PipelineConfig:
    enable_voice_pipeline: bool = False
    interrupt_tts_on_new_input: bool = True
    interrupt_asr_on_new_input: bool = True
    auto_play_tts: bool = True
    emit_asr_text_message: bool = False
    save_audio_cache: bool = True
    audio_cache_dir: str = "desktop_client/data/cache/audio"


@dataclass
class VoiceRuntimeConfig:
    asr: ASRProviderConfig
    tts: TTSProviderConfig
    pipeline: PipelineConfig


def _parse_json_map(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw or "{}")
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def build_runtime_config(voice_cfg: Any) -> VoiceRuntimeConfig:
    asr = ASRProviderConfig(
        enabled=bool(getattr(voice_cfg, "asr_enabled", False)),
        provider_type=str(getattr(voice_cfg, "asr_provider_type", "none") or "none"),
        api_url=str(getattr(voice_cfg, "asr_api_url", "") or ""),
        timeout=int(getattr(voice_cfg, "asr_timeout", 30) or 30),
        upload_field=str(getattr(voice_cfg, "asr_upload_field", "file") or "file"),
        response_text_key=str(getattr(voice_cfg, "asr_response_text_key", "text") or "text"),
        headers_json=str(getattr(voice_cfg, "asr_headers_json", "{}") or "{}"),
        extra_params_json=str(getattr(voice_cfg, "asr_extra_params_json", "{}") or "{}"),
        runtime_backend=str(getattr(voice_cfg, "asr_runtime_backend", "faster_whisper") or "faster_whisper"),
        model_path=str(getattr(voice_cfg, "asr_model_path", "") or ""),
        tokens_path=str(getattr(voice_cfg, "asr_tokens_path", "") or ""),
        device=str(getattr(voice_cfg, "asr_device", "cpu") or "cpu"),
        language=str(getattr(voice_cfg, "asr_language", "zh") or "zh"),
    )

    tts = TTSProviderConfig(
        enabled=bool(getattr(voice_cfg, "tts_enabled", getattr(voice_cfg, "enable_tts", False))),
        provider_type=str(getattr(voice_cfg, "tts_provider_type", "none") or "none"),
        api_url=str(getattr(voice_cfg, "tts_api_url", "") or ""),
        method=str(getattr(voice_cfg, "tts_method", "POST") or "POST").upper(),
        text_field=str(getattr(voice_cfg, "tts_text_field", "text") or "text"),
        response_mode=str(getattr(voice_cfg, "tts_response_mode", "audio_stream") or "audio_stream"),
        response_key=str(getattr(voice_cfg, "tts_response_key", "audio_url") or "audio_url"),
        headers_json=str(getattr(voice_cfg, "tts_headers_json", "{}") or "{}"),
        extra_params_json=str(getattr(voice_cfg, "tts_extra_params_json", "{}") or "{}"),
        audio_format=str(getattr(voice_cfg, "tts_audio_format", "wav") or "wav"),
        runtime_backend=str(getattr(voice_cfg, "tts_runtime_backend", "qt") or "qt"),
        model_path=str(getattr(voice_cfg, "tts_model_path", "") or ""),
        speaker=str(getattr(voice_cfg, "tts_speaker", "") or ""),
        language=str(getattr(voice_cfg, "tts_language", "zh") or "zh"),
    )

    pipe = PipelineConfig(
        enable_voice_pipeline=bool(getattr(voice_cfg, "enable_voice_pipeline", False)),
        interrupt_tts_on_new_input=bool(getattr(voice_cfg, "interrupt_tts_on_new_input", True)),
        interrupt_asr_on_new_input=bool(getattr(voice_cfg, "interrupt_asr_on_new_input", True)),
        auto_play_tts=bool(getattr(voice_cfg, "auto_play_tts", True)),
        emit_asr_text_message=bool(getattr(voice_cfg, "emit_asr_text_message", False)),
        save_audio_cache=bool(getattr(voice_cfg, "save_audio_cache", True)),
        audio_cache_dir=str(getattr(voice_cfg, "audio_cache_dir", "desktop_client/data/cache/audio") or "desktop_client/data/cache/audio"),
    )

    Path(pipe.audio_cache_dir).mkdir(parents=True, exist_ok=True)
    return VoiceRuntimeConfig(asr=asr, tts=tts, pipeline=pipe)


def parse_headers_and_params(cfg: ASRProviderConfig | TTSProviderConfig) -> tuple[Dict[str, Any], Dict[str, Any]]:
    return _parse_json_map(cfg.headers_json), _parse_json_map(cfg.extra_params_json)
