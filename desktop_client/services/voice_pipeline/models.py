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
    provider_type: str = "runtime"  # fixed: runtime
    runtime_backend: str = "genie_tts"  # fixed backend
    mode: str = "predefined"  # predefined / onnx_local
    genie_mode: str = "predefined"
    predefined_character_name: str = ""
    genie_predefined_character_name: str = ""
    genie_predefined_voice: str = ""
    character_name: str = ""
    genie_character_name: str = ""
    onnx_model_dir: str = ""
    genie_model_dir: str = ""
    language: str = "zh"
    genie_language: str = "zh"
    reference_audio_path: str = ""
    genie_reference_audio_path: str = ""
    reference_audio_text: str = ""
    genie_reference_audio_text: str = ""
    timeout: int = 60
    auto_play: bool = True
    save_temp_audio: bool = True
    temp_audio_dir: str = "desktop_client/data/cache/audio"
    use_genie_data_dir: bool = False
    genie_data_dir: str = ""
    gpt_sovits_enabled: bool = False
    gpt_sovits_python_path: str = "python"
    gpt_sovits_api_script_path: str = ""
    gpt_sovits_working_dir: str = ""
    gpt_sovits_host: str = "127.0.0.1"
    gpt_sovits_port: int = 9880
    gpt_sovits_startup_timeout: int = 40
    gpt_sovits_health_timeout: int = 2
    gpt_sovits_request_timeout: int = 90
    gpt_sovits_health_endpoint: str = "/health"
    gpt_sovits_tts_endpoint: str = "/tts"
    gpt_sovits_tts_config_path: str = ""
    gpt_sovits_default_character: str = ""
    gpt_sovits_default_language: str = "zh"
    gpt_sovits_reference_audio_path: str = ""
    gpt_sovits_reference_text: str = ""
    gpt_sovits_segmentation_mode: str = "auto"
    gpt_sovits_segmentation_params_json: str = "{}"
    gpt_sovits_gpt_weights_dir: str = ""
    gpt_sovits_sovits_weights_dir: str = ""
    gpt_sovits_selected_gpt_weights: str = ""
    gpt_sovits_selected_sovits_weights: str = ""
    gpt_sovits_auto_shutdown_on_exit: bool = True


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

    raw_genie_mode = str(getattr(voice_cfg, "genie_mode", "predefined") or "predefined")
    genie_character_name = str(getattr(voice_cfg, "genie_character_name", "") or "")
    genie_model_dir = str(getattr(voice_cfg, "genie_onnx_model_dir", getattr(voice_cfg, "tts_model_path", "")) or "")
    genie_ref_audio = str(getattr(voice_cfg, "genie_reference_audio_path", getattr(voice_cfg, "tts_ref_audio_path", "")) or "")
    genie_ref_text = str(getattr(voice_cfg, "genie_reference_audio_text", getattr(voice_cfg, "tts_prompt_text", "")) or "")
    genie_mode = raw_genie_mode
    if raw_genie_mode == "onnx_local":
        required_ok = bool(genie_character_name and genie_model_dir and genie_ref_audio and genie_ref_text)
        if not required_ok:
            genie_mode = "predefined"

    tts = TTSProviderConfig(
        enabled=bool(getattr(voice_cfg, "enable_local_tts", getattr(voice_cfg, "tts_enabled", getattr(voice_cfg, "enable_tts", False)))),
        provider_type="runtime",
        runtime_backend=str(getattr(voice_cfg, "tts_backend", "genie_tts") or "genie_tts"),
        mode=genie_mode,
        genie_mode=genie_mode,
        predefined_character_name=str(getattr(voice_cfg, "genie_predefined_character_name", "") or ""),
        genie_predefined_character_name=str(getattr(voice_cfg, "genie_predefined_character_name", "") or ""),
        genie_predefined_voice=str(getattr(voice_cfg, "genie_predefined_voice", getattr(voice_cfg, "genie_predefined_character_name", "")) or ""),
        character_name=genie_character_name,
        genie_character_name=genie_character_name,
        onnx_model_dir=genie_model_dir,
        genie_model_dir=genie_model_dir,
        language=str(getattr(voice_cfg, "genie_language", getattr(voice_cfg, "tts_language", "zh")) or "zh"),
        genie_language=str(getattr(voice_cfg, "genie_language", getattr(voice_cfg, "tts_language", "zh")) or "zh"),
        reference_audio_path=genie_ref_audio,
        genie_reference_audio_path=genie_ref_audio,
        reference_audio_text=genie_ref_text,
        genie_reference_audio_text=genie_ref_text,
        timeout=int(getattr(voice_cfg, "genie_timeout", getattr(voice_cfg, "tts_timeout", 60)) or 60),
        auto_play=bool(getattr(voice_cfg, "genie_auto_play", getattr(voice_cfg, "auto_play_tts", True))),
        save_temp_audio=bool(getattr(voice_cfg, "genie_save_temp_audio", True)),
        temp_audio_dir=str(getattr(voice_cfg, "genie_temp_audio_dir", getattr(voice_cfg, "audio_cache_dir", "desktop_client/data/cache/audio")) or "desktop_client/data/cache/audio"),
        use_genie_data_dir=bool(getattr(voice_cfg, "genie_use_data_dir", False)),
        genie_data_dir=str(getattr(voice_cfg, "genie_data_dir", "") or ""),
        gpt_sovits_enabled=bool(getattr(voice_cfg, "gpt_sovits_enabled", False)),
        gpt_sovits_python_path=str(getattr(voice_cfg, "gpt_sovits_python_path", "python") or "python"),
        gpt_sovits_api_script_path=str(getattr(voice_cfg, "gpt_sovits_api_script_path", "") or ""),
        gpt_sovits_working_dir=str(getattr(voice_cfg, "gpt_sovits_working_dir", "") or ""),
        gpt_sovits_host=str(getattr(voice_cfg, "gpt_sovits_host", "127.0.0.1") or "127.0.0.1"),
        gpt_sovits_port=int(getattr(voice_cfg, "gpt_sovits_port", 9880) or 9880),
        gpt_sovits_startup_timeout=int(getattr(voice_cfg, "gpt_sovits_startup_timeout", 40) or 40),
        gpt_sovits_health_timeout=int(getattr(voice_cfg, "gpt_sovits_health_timeout", 2) or 2),
        gpt_sovits_request_timeout=int(getattr(voice_cfg, "gpt_sovits_request_timeout", 90) or 90),
        gpt_sovits_health_endpoint=str(getattr(voice_cfg, "gpt_sovits_health_endpoint", "/health") or "/health"),
        gpt_sovits_tts_endpoint=str(getattr(voice_cfg, "gpt_sovits_tts_endpoint", "/tts") or "/tts"),
        gpt_sovits_tts_config_path=str(getattr(voice_cfg, "gpt_sovits_tts_config_path", "") or ""),
        gpt_sovits_default_character=str(getattr(voice_cfg, "gpt_sovits_default_character", "") or ""),
        gpt_sovits_default_language=str(getattr(voice_cfg, "gpt_sovits_default_language", "zh") or "zh"),
        gpt_sovits_reference_audio_path=str(getattr(voice_cfg, "gpt_sovits_reference_audio_path", "") or ""),
        gpt_sovits_reference_text=str(getattr(voice_cfg, "gpt_sovits_reference_text", "") or ""),
        gpt_sovits_segmentation_mode=str(getattr(voice_cfg, "gpt_sovits_segmentation_mode", "auto") or "auto"),
        gpt_sovits_segmentation_params_json=str(getattr(voice_cfg, "gpt_sovits_segmentation_params_json", "{}") or "{}"),
        gpt_sovits_gpt_weights_dir=str(getattr(voice_cfg, "gpt_sovits_gpt_weights_dir", "") or ""),
        gpt_sovits_sovits_weights_dir=str(getattr(voice_cfg, "gpt_sovits_sovits_weights_dir", "") or ""),
        gpt_sovits_selected_gpt_weights=str(getattr(voice_cfg, "gpt_sovits_selected_gpt_weights", "") or ""),
        gpt_sovits_selected_sovits_weights=str(getattr(voice_cfg, "gpt_sovits_selected_sovits_weights", "") or ""),
        gpt_sovits_auto_shutdown_on_exit=bool(getattr(voice_cfg, "gpt_sovits_auto_shutdown_on_exit", True)),
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
