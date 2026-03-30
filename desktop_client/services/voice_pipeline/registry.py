from __future__ import annotations

from .base import BaseASRProvider, BaseTTSProvider
from .models import ASRProviderConfig, TTSProviderConfig
from .providers.noop import NoopASRProvider, NoopTTSProvider
from .providers.http_asr import HTTPASRProvider
from .providers.http_tts import HTTPTTSProvider
from .providers.runtime_asr import RuntimeASRProvider
from .providers.runtime_tts import RuntimeTTSProvider


def create_asr_provider(config: ASRProviderConfig, logger) -> BaseASRProvider:
    try:
        if not config.enabled or config.provider_type == 'none':
            logger.info('ASR provider: noop')
            return NoopASRProvider()
        if config.provider_type == 'http':
            logger.info(f"ASR provider: http ({config.api_url})")
            return HTTPASRProvider(config, logger)
        if config.provider_type == 'runtime':
            logger.info(f"ASR provider: runtime ({config.runtime_backend})")
            return RuntimeASRProvider(config, logger)
        logger.warning(f"未知 ASR provider_type={config.provider_type}, 回退 noop")
    except Exception as exc:
        logger.error(f"创建 ASR provider 失败，回退 noop: {exc}")
    return NoopASRProvider()


def create_tts_provider(config: TTSProviderConfig, logger, cache_dir: str) -> BaseTTSProvider:
    try:
        if not config.enabled or config.provider_type == 'none':
            logger.info('TTS provider: noop')
            return NoopTTSProvider()
        if config.provider_type == 'http':
            logger.info(f"TTS provider: http ({config.api_url})")
            return HTTPTTSProvider(config, logger, cache_dir=cache_dir)
        if config.provider_type == 'runtime':
            logger.info(f"TTS provider: runtime ({config.runtime_backend})")
            return RuntimeTTSProvider(config, logger, cache_dir=cache_dir)
        logger.warning(f"未知 TTS provider_type={config.provider_type}, 回退 noop")
    except Exception as exc:
        logger.error(f"创建 TTS provider 失败，回退 noop: {exc}")
    return NoopTTSProvider()
