"""
VAD Model Manager - Manages different VAD model implementations
"""

import json
import logging
import os
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Protocol, Callable
import numpy as np
from dataclasses import dataclass

# Import modern i18n module for translations
from . import i18n_modern as i18n

# Convenience imports
_ = i18n._

logger = logging.getLogger(__name__)


@dataclass
class VadConfig:
    """Configuration for VAD models"""
    default_model: str = "whisper_vad"
    auto_inject: bool = False
    ttl: int = 3600  # Cache TTL in seconds

    # VAD parameters
    threshold: float = 0.5
    neg_threshold: Optional[float] = None
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float('inf')
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400

    # ONNX-specific parameters
    onnx_model_path: Optional[str] = None
    onnx_metadata_path: Optional[str] = None
    whisper_model_name: str = "openai/whisper-base"
    frame_duration_ms: int = 20
    chunk_duration_ms: int = 30000
    force_cpu: bool = False
    num_threads: int = 1


class VadModel(Protocol):
    """Protocol for VAD models"""

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get speech timestamps from audio"""
        ...


class WhisperVADOnnxWrapper:
    """ONNX wrapper for Whisper-based VAD model following Silero's architecture."""

    def __init__(
        self,
        model_path: str,
        metadata_path: Optional[str] = None,
        force_cpu: bool = False,
        num_threads: int = 1,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ):
        """Initialize ONNX model wrapper.

        Args:
            model_path: Path to ONNX model file
            metadata_path: Path to metadata JSON file (optional)
            force_cpu: Force CPU execution even if GPU is available
            num_threads: Number of CPU threads for inference
            progress_callback: Optional callback for progress tracking (chunk_idx, total_chunks, device)
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(_("vad.onnx_not_installed"))

        try:
            from transformers import WhisperFeatureExtractor
        except ImportError:
            raise ImportError(_("vad.transformers_not_installed"))

        self.model_path = model_path
        self.progress_callback = progress_callback
        self.device = "CPU"  # Will be updated based on actual provider

        # Load metadata
        if metadata_path is None:
            metadata_path = model_path.replace('.onnx', '_metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            warnings.warn("No metadata file found. Using default values.")
            self.metadata = {
                'whisper_model_name': 'openai/whisper-base',
                'frame_duration_ms': 20,
                'total_duration_ms': 30000,
            }

        # Initialize feature extractor - try local folder first for offline usage
        local_whisper_base_path = Path("models/whisper-base")
        if local_whisper_base_path.exists() and (local_whisper_base_path / "preprocessor_config.json").exists():
            # Load from local folder for offline usage
            try:
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    str(local_whisper_base_path)
                )
                logger.info(_("vad.feature_extractor_loaded", path=local_whisper_base_path))
            except Exception as e:
                warnings.warn(f"Failed to load from local folder, trying online: {e}")
                self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                    self.metadata['whisper_model_name']
                )
        else:
            # Try to load from HuggingFace (requires internet)
            self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
                self.metadata['whisper_model_name']
            )

        # Set up ONNX Runtime session
        opts = ort.SessionOptions()

        # Determine execution provider first
        available_providers = ort.get_available_providers()
        providers = ['CPUExecutionProvider']
        # 优先使用 OpenVINO (Intel GPU), 其次是 CUDA (NVIDIA), 最后是 CPU
        use_openvino = not force_cpu and 'OpenVINOExecutionProvider' in available_providers
        use_cuda = not force_cpu and 'CUDAExecutionProvider' in available_providers and not use_openvino

        if use_openvino:
            providers.insert(0, 'OpenVINOExecutionProvider')
            self.device = "GPU (OpenVINO)"
            logger.info("✅ VAD will use Intel GPU (OpenVINO Execution Provider).")
            # OpenVINO 通常自己管理线程，可以不设置
        elif use_cuda:
            providers.insert(0, 'CUDAExecutionProvider')
            self.device = "GPU (CUDA)"
            logger.info("✅ VAD will use NVIDIA GPU (CUDA Execution Provider).")
            opts.inter_op_num_threads = num_threads
            opts.intra_op_num_threads = num_threads
        else:
            self.device = "CPU"
            logger.info("⚠️ VAD will use CPU.")
            # For CPU, optimize thread count
            import multiprocessing
            if num_threads == 1:
                optimal_threads = max(1, multiprocessing.cpu_count() // 2)
                opts.inter_op_num_threads = optimal_threads
                opts.intra_op_num_threads = optimal_threads
                logger.info(f"Auto-configured VAD CPU threads to {optimal_threads} (Total: {multiprocessing.cpu_count()})")
            else:
                opts.inter_op_num_threads = num_threads
                opts.intra_op_num_threads = num_threads

        try:
            # 创建一个与 providers 列表长度相匹配的 provider_options 列表
            # 默认为每个 provider 提供一个空选项
            provider_options = [{} for _ in providers]

            # 如果 OpenVINO 可用，找到它的位置并设置它的专属选项
            if 'OpenVINOExecutionProvider' in providers:
                index = providers.index('OpenVINOExecutionProvider')
                provider_options[index] = {'device_type': 'GPU_FP16'}
                logger.info("已为OpenVINOExecutionProvider强制配置 'device_type': 'GPU_FP16'")

            self.session = ort.InferenceSession(
                model_path,
                providers=providers,
                provider_options=provider_options, # <-- 现在两个列表长度匹配了
                sess_options=opts
            )
            logger.info("✅ ONNX Session成功创建，Provider选项配置合法。")

        except Exception as e:
            # 如果强制使用GPU失败，这里会捕获到具体的错误信息
            logger.error(f"❌ 创建ONNX Session失败！这通常意味着OpenVINO无法在你的GPU上运行。")
            logger.error(f"   具体错误: {e}")
            logger.info("   正在尝试回退到仅使用CPU...")

            # 尝试仅使用CPU来创建Session，确保程序还能运行
            try:
                self.session = ort.InferenceSession(
                    model_path,
                    providers=['CPUExecutionProvider'],
                    sess_options=opts
                )
                self.device = "CPU"
                logger.info("✅ 成功回退到CPU模式创建Session。")
            except Exception as cpu_e:
                logger.error(f"❌ 连CPU模式都失败了！错误: {cpu_e}")
                raise cpu_e

        # Get input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        # Model parameters
        self.sample_rate = 16000  # Whisper uses 16kHz
        self.frame_duration_ms = self.metadata.get('frame_duration_ms', 20)
        self.chunk_duration_ms = self.metadata.get('total_duration_ms', 30000)
        self.chunk_samples = int(self.chunk_duration_ms * self.sample_rate / 1000)
        self.frames_per_chunk = int(self.chunk_duration_ms / self.frame_duration_ms)

        # Initialize state
        self.reset_states()

        logger.info(_("vad.model_loaded", path=model_path))
        logger.info(_("vad.device", device=self.device))
        logger.info(_("vad.providers", providers=providers))
        logger.info(_("vad.chunk_duration", duration=self.chunk_duration_ms))
        logger.info(_("vad.frame_duration", duration=self.frame_duration_ms))

    def reset_states(self):
        """Reset internal states for new audio stream."""
        self._context = None
        self._last_chunk = None

    def _validate_input(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Validate and preprocess input audio.

        Args:
            audio: Input audio array
            sr: Sample rate

        Returns:
            Preprocessed audio at 16kHz
        """
        if audio.ndim > 1:
            # Convert to mono if multi-channel
            audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

        # Resample if needed
        if sr != self.sample_rate:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            except ImportError:
                logger.warning(_("vad.librosa_not_installed"))
                # Basic downsampling if librosa not available
                if sr > self.sample_rate:
                    # Simple downsampling
                    ratio = sr // self.sample_rate
                    audio = audio[::ratio]

        return audio

    def __call__(self, audio_chunk: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process a single audio chunk.

        Args:
            audio_chunk: Audio chunk to process
            sr: Sample rate

        Returns:
            Frame-level speech probabilities
        """
        # Validate input
        audio_chunk = self._validate_input(audio_chunk, sr)

        # Ensure chunk is correct size
        if len(audio_chunk) < self.chunk_samples:
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.chunk_samples - len(audio_chunk)),
                mode='constant'
            )
        elif len(audio_chunk) > self.chunk_samples:
            audio_chunk = audio_chunk[:self.chunk_samples]

        # Extract features
        inputs = self.feature_extractor(
            audio_chunk,
            sampling_rate=self.sample_rate,
            return_tensors="np"
        )

        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: inputs.input_features}
        )

        # Apply sigmoid to get probabilities
        frame_logits = outputs[0][0]  # Remove batch dimension
        frame_probs = 1 / (1 + np.exp(-frame_logits))

        return frame_probs

    def audio_forward(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Process full audio file in chunks (Silero-style).

        Args:
            audio: Full audio array
            sr: Sample rate

        Returns:
            Concatenated frame probabilities for entire audio
        """
        audio = self._validate_input(audio, sr)
        self.reset_states()

        all_probs = []

        # Calculate total number of chunks
        total_chunks = (len(audio) + self.chunk_samples - 1) // self.chunk_samples

        # Log initial processing info
        logger.info(_("vad.starting", device=self.device))
        logger.info(_("vad.total_samples", samples=len(audio)))
        logger.info(_("vad.chunk_size", samples=self.chunk_samples, duration=self.chunk_duration_ms))
        logger.info(_("vad.total_chunks", chunks=total_chunks))

        # Process in chunks
        for chunk_idx, i in enumerate(range(0, len(audio), self.chunk_samples)):
            chunk = audio[i:i + self.chunk_samples]

            # Pad last chunk if needed
            if len(chunk) < self.chunk_samples:
                chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)), mode='constant')

            # Report progress
            if self.progress_callback:
                self.progress_callback(chunk_idx + 1, total_chunks, self.device)

            # Log chunk progress
            progress_pct = ((chunk_idx + 1) / total_chunks) * 100
            logger.debug(_("vad.processing_chunk", current=chunk_idx + 1, total=total_chunks,
                percent=progress_pct, device=self.device))

            # Get predictions for chunk
            chunk_probs = self.__call__(chunk, self.sample_rate)
            all_probs.append(chunk_probs)

        logger.info(_("vad.completed", chunks=total_chunks, device=self.device))

        # Concatenate all probabilities
        if all_probs:
            return np.concatenate(all_probs)
        return np.array([])


def get_speech_timestamps_onnx(
    audio: np.ndarray,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    max_speech_duration_s: float = float('inf'),
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = True,
    neg_threshold: Optional[float] = None,
    progress_tracking_callback: Optional[Callable[[float], None]] = None,
) -> List[Dict[str, float]]:
    """Extract speech timestamps from audio using Silero-style processing.

    This function implements Silero VAD's approach with:
    - Dual threshold (positive and negative) for hysteresis
    - Proper segment padding
    - Minimum duration filtering
    - Maximum duration handling with intelligent splitting

    Args:
        audio: Input audio array
        model: VAD model (WhisperVADOnnxWrapper instance)
        threshold: Speech threshold (default: 0.5)
        sampling_rate: Audio sample rate
        min_speech_duration_ms: Minimum speech segment duration
        max_speech_duration_s: Maximum speech segment duration
        min_silence_duration_ms: Minimum silence to split segments
        speech_pad_ms: Padding to add to speech segments
        return_seconds: Return times in seconds vs samples
        neg_threshold: Negative threshold for hysteresis (default: threshold - 0.15)
        progress_tracking_callback: Progress callback function

    Returns:
        List of speech segments with start/end times
    """
    # Audio should already be numpy array

    # Validate audio
    if audio.ndim > 1:
        audio = audio.mean(axis=0 if audio.shape[0] > audio.shape[1] else 1)

    # Get frame probabilities for entire audio
    model.reset_states()
    speech_probs = model.audio_forward(audio, sampling_rate)

    # Calculate frame parameters
    frame_duration_ms = model.frame_duration_ms
    frame_samples = int(sampling_rate * frame_duration_ms / 1000)

    # Convert durations to frames
    min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
    min_silence_frames = int(min_silence_duration_ms / frame_duration_ms)
    speech_pad_frames = int(speech_pad_ms / frame_duration_ms)
    max_speech_frames = int(max_speech_duration_s * 1000 / frame_duration_ms) if max_speech_duration_s != float('inf') else len(speech_probs)

    # Set negative threshold for hysteresis
    if neg_threshold is None:
        neg_threshold = max(threshold - 0.15, 0.01)

    # Track speech segments
    triggered = False
    speeches = []
    current_speech = {}
    current_probs = []  # Track probabilities for current segment
    temp_end = 0

    # Process each frame
    for i, speech_prob in enumerate(speech_probs):
        # Report progress
        if progress_tracking_callback:
            progress = (i + 1) / len(speech_probs) * 100
            progress_tracking_callback(progress)

        # Track probabilities for current segment
        if triggered:
            current_probs.append(float(speech_prob))

        # Speech onset detection
        if speech_prob >= threshold and not triggered:
            triggered = True
            current_speech['start'] = i
            current_probs = [float(speech_prob)]  # Start tracking probabilities
            continue

        # Check for maximum speech duration
        if triggered and 'start' in current_speech:
            duration = i - current_speech['start']
            if duration > max_speech_frames:
                # Force end segment at max duration
                current_speech['end'] = current_speech['start'] + max_speech_frames
                # Calculate probability statistics for segment
                if current_probs:
                    current_speech['probability'] = np.mean(current_probs)
                speeches.append(current_speech)
                current_speech = {}
                current_probs = []
                triggered = False
                temp_end = 0
                continue

        # Speech offset detection with hysteresis
        if speech_prob < neg_threshold and triggered:
            if not temp_end:
                temp_end = i

            # Check if silence is long enough
            if i - temp_end >= min_silence_frames:
                # End current speech segment
                current_speech['end'] = temp_end

                # Check minimum duration
                if current_speech['end'] - current_speech['start'] >= min_speech_frames:
                    # Calculate probability statistics for segment
                    if current_probs:
                        current_speech['probability'] = np.mean(current_probs[:temp_end - current_speech['start']])
                    speeches.append(current_speech)

                current_speech = {}
                current_probs = []
                triggered = False
                temp_end = 0

        # Reset temp_end if speech resumes
        elif speech_prob >= threshold and temp_end:
            temp_end = 0

    # Handle speech that continues to the end
    if triggered and 'start' in current_speech:
        current_speech['end'] = len(speech_probs)
        if current_speech['end'] - current_speech['start'] >= min_speech_frames:
            # Calculate probability statistics for segment
            if current_probs:
                current_speech['probability'] = np.mean(current_probs)
            speeches.append(current_speech)

    # Apply padding to segments
    for i, speech in enumerate(speeches):
        # Add padding
        if i == 0:
            speech['start'] = max(0, speech['start'] - speech_pad_frames)
        else:
            speech['start'] = max(speeches[i-1]['end'], speech['start'] - speech_pad_frames)

        if i < len(speeches) - 1:
            speech['end'] = min(speeches[i+1]['start'], speech['end'] + speech_pad_frames)
        else:
            speech['end'] = min(len(speech_probs), speech['end'] + speech_pad_frames)

    # Convert to time units or sample indices based on return_seconds
    for speech in speeches:
        if return_seconds:
            # Convert frame indices to seconds
            speech['start'] = speech['start'] * frame_duration_ms / 1000
            speech['end'] = speech['end'] * frame_duration_ms / 1000
        else:
            # Convert frame indices to sample indices
            speech['start'] = int(speech['start'] * frame_samples)
            speech['end'] = int(speech['end'] * frame_samples)

    return speeches


class WhisperVadModel:
    """
    Whisper-based VAD model implementation using ONNX.
    Uses a Whisper model exported to ONNX for voice activity detection.
    """

    def __init__(self, config: Optional[VadConfig] = None, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        self.config = config or VadConfig()
        self.wrapper = None
        self.progress_callback = progress_callback

        # Initialize ONNX model if path provided
        if self.config.onnx_model_path and os.path.exists(self.config.onnx_model_path):
            try:
                self.wrapper = WhisperVADOnnxWrapper(
                    model_path=self.config.onnx_model_path,
                    metadata_path=self.config.onnx_metadata_path,
                    force_cpu=self.config.force_cpu,
                    num_threads=self.config.num_threads,
                    progress_callback=progress_callback,
                )
                logger.info(_("vad.model_initialized", path=self.config.onnx_model_path))
                if self.wrapper.device:
                    logger.info(_("vad.using_device", device=self.wrapper.device))
            except Exception as e:
                logger.error(_("vad.init_failed", error=e))
        else:
            logger.warning(_("vad.path_invalid", path=self.config.onnx_model_path))

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        threshold: float = None,
        min_speech_duration_ms: int = None,
        max_speech_duration_s: float = None,
        min_silence_duration_ms: int = None,
        speech_pad_ms: int = None,
        neg_threshold: float = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get speech timestamps using Whisper VAD.
        """
        if self.wrapper is None:
            logger.error(_("vad.not_initialized"))
            return []

        # Use provided parameters or defaults from config
        threshold = threshold if threshold is not None else self.config.threshold
        neg_threshold = neg_threshold if neg_threshold is not None else self.config.neg_threshold
        min_speech_duration_ms = min_speech_duration_ms if min_speech_duration_ms is not None else self.config.min_speech_duration_ms
        max_speech_duration_s = max_speech_duration_s if max_speech_duration_s is not None else self.config.max_speech_duration_s
        min_silence_duration_ms = min_silence_duration_ms if min_silence_duration_ms is not None else self.config.min_silence_duration_ms
        speech_pad_ms = speech_pad_ms if speech_pad_ms is not None else self.config.speech_pad_ms

        # Use ONNX model for speech detection
        # Return sample indices (not seconds) for compatibility with faster_whisper
        segments = get_speech_timestamps_onnx(
            audio=audio,
            model=self.wrapper,
            threshold=threshold,
            sampling_rate=sampling_rate,
            min_speech_duration_ms=min_speech_duration_ms,
            max_speech_duration_s=max_speech_duration_s,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=False,  # faster_whisper expects sample indices
            neg_threshold=neg_threshold,
        )

        logger.debug(_("vad.speech_segments", count=len(segments)))
        return segments

    def get_device(self) -> str:
        """Get the device being used for VAD processing."""
        if self.wrapper:
            return self.wrapper.device
        return "Not initialized"


class VadModelManager:
    """
    Manages different VAD model implementations.
    Provides a unified interface for VAD operations.
    """

    def __init__(self, config: Optional[VadConfig] = None, ttl: int = 3600, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        self.config = config or VadConfig()
        self.ttl = ttl
        self.progress_callback = progress_callback
        self._models: Dict[str, VadModel] = {}  # Instance variable, not class variable
        self._config = config

        # Register available models
        self._register_models()

    def _register_models(self):
        """Register available VAD models"""
        # Always recreate the model with the current progress callback
        self._models["whisper_vad"] = WhisperVadModel(self.config, progress_callback=self.progress_callback)
        logger.debug(_("vad.registered"))

    def get_model(self, model_id: str) -> VadModel:
        """Get a VAD model by ID"""
        if model_id not in self._models:
            logger.warning(_("vad.model_not_found", model_id=model_id))
            model_id = self.config.default_model

        return self._models.get(model_id, self._models["whisper_vad"])

    def get_speech_timestamps(
        self,
        model_id: str,
        audio: np.ndarray,
        sampling_rate: int = 16000,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get speech timestamps using specified model.

        Args:
            model_id: ID of the VAD model to use
            audio: Audio array
            sampling_rate: Sample rate of audio
            **kwargs: Additional parameters for the VAD model

        Returns:
            List of speech segments with start, end, and probability
        """
        model = self.get_model(model_id)
        return model.get_speech_timestamps(audio, sampling_rate, **kwargs)

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available VAD models"""
        # Since models are now instance variables, return the known model types
        return ["whisper_vad"]

    def get_device(self, model_id: str = None) -> str:
        """Get the device being used for VAD processing."""
        if model_id is None:
            model_id = self.config.default_model
        model = self.get_model(model_id)
        if hasattr(model, 'get_device'):
            return model.get_device()
        return "Unknown"