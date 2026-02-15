"""
VAD Injection System - Redirects faster_whisper VAD calls to custom implementations
Provides transparent switching between custom VAD models
"""

import unittest.mock as mock
from typing import List, Dict, Any, Optional, Callable
import logging
import numpy as np
from dataclasses import dataclass

from .vad_manager import VadModelManager, VadConfig

# Import modern i18n module for translations
from . import i18n_modern as i18n

# Convenience imports
_ = i18n._

logger = logging.getLogger(__name__)

# Global flag to track if injection is active
_injection_active = False
_active_patches = []
_global_config = None
_global_progress_callback = None


@dataclass
class VadOptionsCompat:
    """Mock VadOptions class that mimics faster_whisper.vad.VadOptions"""
    threshold: float = 0.5
    neg_threshold: Optional[float] = None
    min_speech_duration_ms: int = 0
    max_speech_duration_s: float = float('inf')
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400

    def __post_init__(self):
        """Compatibility with the original VadOptions"""
        pass


def set_global_config(config: VadConfig):
    """Set the global configuration for VAD injection"""
    global _global_config
    _global_config = config


def get_global_config() -> VadConfig:
    """Get the global configuration, creating default if needed"""
    global _global_config
    if _global_config is None:
        _global_config = VadConfig()
    return _global_config


def get_speech_timestamps_injected(
    audio: np.ndarray,
    vad_options: Any = None,
    sampling_rate: int = 16000,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Injected implementation of get_speech_timestamps that uses our VAD model manager.

    This function is injected in place of faster_whisper.vad.get_speech_timestamps
    to transparently use custom VAD models.
    """
    # Get configuration
    config = get_global_config()

    # Check if a specific model was requested via kwargs
    model_id = kwargs.get('vad_model_id', config.default_model)

    # Check if a progress callback was provided (from kwargs or global)
    progress_callback = kwargs.get('progress_callback', None) or _global_progress_callback

    # Create manager (this uses cached instances internally)
    manager = VadModelManager(config=config, ttl=config.ttl, progress_callback=progress_callback)

    # Extract options from vad_options (works with both real and mock VadOptions)
    if vad_options is not None:
        options_dict = {
            'threshold': getattr(vad_options, 'threshold', config.threshold),
            'neg_threshold': getattr(vad_options, 'neg_threshold', config.neg_threshold),
            'min_speech_duration_ms': getattr(vad_options, 'min_speech_duration_ms', config.min_speech_duration_ms),
            'max_speech_duration_s': getattr(vad_options, 'max_speech_duration_s', config.max_speech_duration_s),
            'min_silence_duration_ms': getattr(vad_options, 'min_silence_duration_ms', config.min_silence_duration_ms),
            'speech_pad_ms': getattr(vad_options, 'speech_pad_ms', config.speech_pad_ms),
        }
    else:
        # Use defaults from config
        options_dict = {
            'threshold': config.threshold,
            'neg_threshold': config.neg_threshold,
            'min_speech_duration_ms': config.min_speech_duration_ms,
            'max_speech_duration_s': config.max_speech_duration_s,
            'min_silence_duration_ms': config.min_silence_duration_ms,
            'speech_pad_ms': config.speech_pad_ms,
        }

    # Remove vad_model_id and progress_callback from kwargs to avoid passing them to the actual VAD
    kwargs_copy = kwargs.copy()
    kwargs_copy.pop('vad_model_id', None)
    kwargs_copy.pop('progress_callback', None)

    # Merge options_dict with remaining kwargs
    final_kwargs = {**options_dict, **kwargs_copy}

    # Get speech timestamps using the model manager
    return manager.get_speech_timestamps(
        model_id=model_id,
        audio=audio,
        sampling_rate=sampling_rate,
        **final_kwargs
    )


def get_vad_patches(model_id: Optional[str] = None) -> Dict[str, mock.Mock]:
    """
    Get all VAD-related patches for the codebase.

    Args:
        model_id: Optional model ID to force (e.g., "whisper_vad")

    Returns:
        Dictionary of patch paths to mock objects
    """
    # Create wrapper functions that include model_id if specified
    if model_id:
        def get_timestamps_wrapper(audio, vad_options=None, sampling_rate=16000, **kwargs):
            kwargs['vad_model_id'] = model_id
            return get_speech_timestamps_injected(audio, vad_options, sampling_rate, **kwargs)
    else:
        get_timestamps_wrapper = get_speech_timestamps_injected

    patches = {
        # Core VAD module patches
        'faster_whisper.vad.VadOptions': mock.Mock(side_effect=VadOptionsCompat),
        'faster_whisper.vad.get_speech_timestamps': mock.Mock(side_effect=get_timestamps_wrapper),

        # Alternative import location (used in transcribe module)
        'faster_whisper.transcribe.get_speech_timestamps': mock.Mock(side_effect=get_timestamps_wrapper),

        # Patch for VadOptions in transcribe module
        'faster_whisper.transcribe.VadOptions': mock.Mock(side_effect=VadOptionsCompat),

        # You can add more patches here for specific modules if needed
        # For example, if you have modules that directly import from faster_whisper:
        # 'your_module.VadOptions': mock.Mock(side_effect=VadOptionsCompat),
        # 'your_module.get_speech_timestamps': mock.Mock(side_effect=get_timestamps_wrapper),
    }

    return patches


def inject_vad(model_id: Optional[str] = None, config: Optional[VadConfig] = None, progress_callback: Optional[Callable] = None) -> None:
    """
    Inject VAD implementation to redirect faster_whisper calls.

    Args:
        model_id: Optional model ID to force (e.g., "whisper_vad")
                 If None, uses the configured default model.
        config: Optional VadConfig to use for injection
        progress_callback: Optional progress callback for VAD processing
    """
    global _injection_active, _active_patches, _global_progress_callback

    if _injection_active:
        logger.warning(_("injection.already_active"))
        return

    # Store progress callback globally
    _global_progress_callback = progress_callback

    # Set config if provided
    if config:
        set_global_config(config)

    patches_dict = get_vad_patches(model_id)

    for path, mock_obj in patches_dict.items():
        try:
            patch = mock.patch(path, mock_obj)
            patch.start()
            _active_patches.append(patch)
            logger.debug(_("injection.patched", path=path))
        except Exception as e:
            logger.debug(_("injection.patch_failed", path=path, error=e))

    _injection_active = True
    if model_id:
        logger.info(_("injection.activated_with_model", model_id=model_id))
    else:
        logger.info(_("injection.activated"))


def uninject_vad() -> None:
    """
    Remove VAD injection and restore original faster_whisper behavior.
    """
    global _injection_active, _active_patches, _global_progress_callback

    if not _injection_active:
        logger.warning(_("injection.not_active"))
        return

    for patch in _active_patches:
        try:
            patch.stop()
        except Exception as e:
            logger.warning(_("injection.stop_error", error=e))

    _active_patches.clear()
    _injection_active = False
    _global_progress_callback = None  # Clear the progress callback
    logger.info(_("info.vad_deactivated"))


class VadInjectionContext:
    """
    Context manager for VAD injection.

    Usage:
        with VadInjectionContext(model_id="whisper_vad"):
            # Code that uses faster_whisper VAD will now use whisper VAD
            from faster_whisper.vad import get_speech_timestamps
            timestamps = get_speech_timestamps(audio, vad_options)
    """

    def __init__(self, model_id: Optional[str] = None, config: Optional[VadConfig] = None):
        self.model_id = model_id
        self.config = config
        self.was_active = False

    def __enter__(self):
        global _injection_active
        self.was_active = _injection_active
        if self.was_active:
            uninject_vad()
        inject_vad(self.model_id, self.config)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        uninject_vad()
        if self.was_active:
            inject_vad()  # Restore previous injection


def auto_inject_vad(config: Optional[VadConfig] = None) -> None:
    """
    Automatically inject VAD based on configuration.
    This should be called during application startup.

    Args:
        config: Optional VadConfig to use
    """
    if config is None:
        config = get_global_config()
    else:
        set_global_config(config)

    # Check if we should inject based on configuration
    if config.auto_inject:
        model_id = config.default_model
        inject_vad(model_id, config)
        logger.info(_("injection.auto_injected", model_id=model_id))


def with_vad_injection(model_id: Optional[str] = None, config: Optional[VadConfig] = None):
    """
    Decorator to use VAD injection for a specific function.

    Usage:
        @with_vad_injection(model_id="whisper_vad")
        def my_function():
            # This function will use whisper VAD
            from faster_whisper.vad import get_speech_timestamps
            return get_speech_timestamps(audio, vad_options)
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            with VadInjectionContext(model_id, config):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def is_injection_active() -> bool:
    """Check if VAD injection is currently active"""
    return _injection_active