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
    encoder_path: str = ""
    decoder_path: str = ""
    joiner_path: str = ""
    device: str = "cpu"
    language: str = "zh"


@dataclass
class TTSProviderConfig:
    enabled: bool = False
    provider_type: str = "none"  # none / http / runtime
    api_url: str = ""
    method: str = "POST"  # 兼容 http provider
    text_field: str = "text"  # 兼容 http provider
    response_mode: str = "audio_stream"  # 兼容 http provider
    response_key: str = "audio_url"  # 兼容 http provider
    runtime_backend: str = "qt"  # qt / pyttsx3 / edge_tts / gpt_sovits / custom
    model_path: str = ""
    speaker: str = ""
    language: str = "zh"
    ref_audio_path: str = ""
    prompt_text: str = ""
    prompt_lang: str = ""
    timeout: int = 60
    headers_json: str = "{}"  # 兼容旧配置：仅用于 HTTP headers
    extra_params_json: str = "{}"  # 兼容 http provider
    audio_format: str = "wav"


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
    raw_backend = str(getattr(voice_cfg, "asr_runtime_backend", "") or "").strip()
    auto_sherpa = bool(
        getattr(voice_cfg, "asr_model_path", "")
        or getattr(voice_cfg, "asr_tokens_path", "")
        or getattr(voice_cfg, "asr_encoder_path", "")
        or getattr(voice_cfg, "asr_decoder_path", "")
        or getattr(voice_cfg, "asr_joiner_path", "")
    )
    asr_backend = raw_backend or ("sherpa_onnx" if auto_sherpa else "faster_whisper")

    asr = ASRProviderConfig(
        enabled=bool(getattr(voice_cfg, "asr_enabled", False)),
        provider_type=str(getattr(voice_cfg, "asr_provider_type", "none") or "none"),
        api_url=str(getattr(voice_cfg, "asr_api_url", "") or ""),
        timeout=int(getattr(voice_cfg, "asr_timeout", 30) or 30),
        upload_field=str(getattr(voice_cfg, "asr_upload_field", "file") or "file"),
        response_text_key=str(getattr(voice_cfg, "asr_response_text_key", "text") or "text"),
        headers_json=str(getattr(voice_cfg, "asr_headers_json", "{}") or "{}"),
        extra_params_json=str(getattr(voice_cfg, "asr_extra_params_json", "{}") or "{}"),
        runtime_backend=asr_backend,
        model_path=str(getattr(voice_cfg, "asr_model_path", "") or ""),
        tokens_path=str(getattr(voice_cfg, "asr_tokens_path", "") or ""),
        encoder_path=str(getattr(voice_cfg, "asr_encoder_path", "") or ""),
        decoder_path=str(getattr(voice_cfg, "asr_decoder_path", "") or ""),
        joiner_path=str(getattr(voice_cfg, "asr_joiner_path", "") or ""),
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
        runtime_backend=str(getattr(voice_cfg, "tts_runtime_backend", "qt") or "qt"),
        model_path=str(getattr(voice_cfg, "tts_model_path", "") or ""),
        speaker=str(getattr(voice_cfg, "tts_speaker", "") or ""),
        language=str(getattr(voice_cfg, "tts_language", "zh") or "zh"),
        ref_audio_path=str(getattr(voice_cfg, "tts_ref_audio_path", "") or ""),
        prompt_text=str(getattr(voice_cfg, "tts_prompt_text", "") or ""),
        prompt_lang=str(getattr(voice_cfg, "tts_prompt_lang", "") or ""),
        timeout=int(getattr(voice_cfg, "tts_timeout", 60) or 60),
        headers_json=str(getattr(voice_cfg, "tts_headers_json", "{}") or "{}"),
        extra_params_json=str(getattr(voice_cfg, "tts_extra_params_json", "{}") or "{}"),
        audio_format=str(getattr(voice_cfg, "tts_audio_format", "wav") or "wav"),
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
    headers_raw = getattr(cfg, "headers_json", "{}")
    params_raw = getattr(cfg, "extra_params_json", "{}")
    return _parse_json_map(headers_raw), _parse_json_map(params_raw)
