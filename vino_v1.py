# colab_batch_processor_auto_continuous.py
import os
import sys
import logging
import shutil
import time
from pathlib import Path
from faster_whisper import WhisperModel
import argparse
# <<< 变化开始 >>>
# 不再使用 faster_whisper 的 WhisperModel
# from faster_whisper import WhisperModel
# import ctranslate2
import torch
import librosa # 使用 librosa 加载音频
from transformers import WhisperProcessor
from optimum.intel import OVModelForSpeechSeq2Seq
from src.faster_whisper_transwithai_chickenrice.injection import VadOptionsCompat
# <<< 变化结束 >>>

try:
    import openvino
    # 获取 openvino.libs 文件夹的绝对路径
    openvino_libs_path = os.path.abspath(os.path.join(os.path.dirname(openvino.runtime.__file__), "..", "libs"))
    # 将这个路径动态添加到 DLL 的搜索路径中
    os.add_dll_directory(openvino_libs_path)
    print(f"成功将OpenVINO DLL路径添加到运行时: {openvino_libs_path}")
except Exception as e:
    print(f"警告: 无法自动添加OpenVINO DLL路径。错误: {e}")


# 项目路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT)) # 确保路径是字符串

# 引入 VAD 注入模块
from src.faster_whisper_transwithai_chickenrice.injection import inject_vad, uninject_vad
from src.faster_whisper_transwithai_chickenrice.vad_manager import VadConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def my_progress_callback(chunk_idx, total_chunks, device):
    """自定义的VAD进度回调函数"""
    progress_pct = (chunk_idx / total_chunks) * 100
    print(f"\r  [VAD] 正在处理音频块: {chunk_idx}/{total_chunks} ({progress_pct:.2f}%) on {device}", end="", flush=True)
    if chunk_idx == total_chunks:
        print()


class ContinuousFolderProcessor:
    """
    连续文件夹处理器（修复版）：
    1. 检查audio文件夹是否有待处理文件
    2. 处理audio文件夹中的所有音频
    3. 处理完成后，将整个文件夹移动到audio_ok
    4. 从audio1中拉取下一个子文件夹到audio
    5. 继续处理下一个文件夹，直到所有文件夹处理完成
    """
    
    def __init__(self, 
                 audio_not_dir=None,
                 audio1_dir=None,
                 audio_dir=None,
                 audio_ok_dir=None,
                 output_dir=None,
                 model_path=None,
                 device="GPU",
                 compute_type="FP16", 
                 use_batch=False,
                 batch_size=8,
                 enable_segment_merge=False):
        
        if audio_not_dir is None: audio_not_dir = PROJECT_ROOT / "audio_not"
        if audio1_dir is None: audio1_dir = PROJECT_ROOT / "audio1"
        if audio_dir is None: audio_dir = PROJECT_ROOT / "audio"
        if audio_ok_dir is None: audio_ok_dir = PROJECT_ROOT / "audio_ok"
        if output_dir is None: output_dir = PROJECT_ROOT / "sub"
        
        self.audio_not_dir = Path(audio_not_dir)
        self.audio1_dir = Path(audio1_dir)
        self.audio_dir = Path(audio_dir)
        self.audio_ok_dir = Path(audio_ok_dir)
        self.output_dir = Path(output_dir)
        self.error_dir = Path(audio_dir).parent / "audio_error"
        
        self.use_batch = use_batch
        self.batch_size = batch_size
        self.enable_segment_merge = enable_segment_merge

        
        self.audio_not_dir.mkdir(parents=True, exist_ok=True)
        self.audio1_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_ok_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)
        
        if model_path is None:
            # <<< 变化：指向转换后的OpenVINO模型路径
            model_path = str(PROJECT_ROOT / "models" / "whisper-large-v2-ov")
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type # 现在主要用于日志记录
        self.model = None
        self.processor = None
        
        self.processed_folders = []
        self.failed_folders = []
        self.error_files = []
        
        self.generate_config = {
            "language": "ja",
            "task": "translate",
            "max_initial_timestamp": 30,
            "repetition_penalty": 1.1,
            "return_timestamps": True # 关键：让模型返回时间戳
        }
        
        logger.info("=" * 60)
        logger.info("连续文件夹处理器初始化完成（支持高优先级audio_not）")
        logger.info(f"高优先级池: {self.audio_not_dir}")
        logger.info(f"低优先级池: {self.audio1_dir}")
        logger.info(f"当前处理: {self.audio_dir}")
        logger.info(f"完成目录: {self.audio_ok_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"错误目录: {self.error_dir}")
        if self.use_batch:
            logger.info(f"批处理模式: 启用 (batch_size={self.batch_size})")
        else:
            logger.info(f"批处理模式: 禁用")
        logger.info("=" * 60)
    
    def setup_vad(self):
        """设置VAD注入"""
        logger.info("初始化 VAD ...")
        uninject_vad()
        logger.info("已清理历史VAD注入")
        
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if 'OpenVINOExecutionProvider' in providers:
                use_gpu = True
                logger.info("✅ 检测到 Intel OpenVINO，VAD将使用Intel GPU加速")
                
                # 尝试设置新的 OpenVINO 配置以避免弃用警告
                try:
                    import onnxruntime as ort
                    # 新的配置方式：设备类型和精度分开
                    session_options = ort.SessionOptions()
                    # 尝试使用新版配置
                    provider_options = {
                        'device_type': 'GPU',  # 新版本不再用 GPU_FP16
                        'precision': 'float16',   # 单独设置精度
                        # 'device_id': '0',    # 如果有多个GPU可以指定
                    }
                    # 测试是否可以创建会话
                    test_session = ort.InferenceSession(
                        "dummy_path",  # 只是测试配置
                        providers=[('OpenVINOExecutionProvider', provider_options)],
                        sess_options=session_options
                    )
                    logger.info("✅ 已配置新版 OpenVINO 参数")
                except:
                    # 如果新版配置失败，回退到旧版
                    logger.info("ℹ️ 使用兼容模式 OpenVINO 配置")
            elif 'CUDAExecutionProvider' in providers:
                use_gpu = True
                logger.info("✅ 检测到 NVIDIA CUDA，VAD将使用NVIDIA GPU加速")
            else:
                use_gpu = False
                logger.info("⚠️ 未检测到可用GPU后端，VAD将使用CPU")
        except Exception as e:
            use_gpu = False
            logger.warning(f"⚠️ 无法检测GPU，VAD将使用CPU。错误: {e}")
        
        
        cfg = VadConfig()
        cfg.onnx_model_path = str(PROJECT_ROOT / "models" / "whisper_vad.onnx")
        cfg.onnx_metadata_path = str(PROJECT_ROOT / "models" / "whisper_vad_metadata.json")
        cfg.threshold = 0.5
        cfg.min_speech_duration_ms = 300
        cfg.min_silence_duration_ms = 100
        cfg.speech_pad_ms = 200
        
        cfg.force_cpu = not use_gpu
        cfg.num_threads = 1 if use_gpu else 20
        
        inject_vad("whisper_vad", cfg, progress_callback=my_progress_callback)
        logger.info("✓ VAD 注入完成")
    
    # <<< 变化开始：重写 load_model >>>
    def load_model(self):
        """加载Whisper OpenVINO模型"""
        logger.info(f"正在为 Whisper OpenVINO 模型配置: 设备='{self.device}'")
        logger.info(f"加载模型: {self.model_path}")
        
        try:
            # OVModelForSpeechSeq2Seq 需要一个 device 参数，例如 "CPU", "GPU", "AUTO"
            # "GPU" 会特指Intel的 iGPU 或 dGPU
            self.model = OVModelForSpeechSeq2Seq.from_pretrained(
                self.model_path,
                device=self.device.upper(), # 确保是大写
                ov_config={"PERFORMANCE_HINT": "LATENCY"}, # 针对单个文件处理进行优化
                compile=True # 加载时编译模型
            )
            self.processor = WhisperProcessor.from_pretrained(self.model_path)
            
            logger.info(f"✓ OpenVINO 模型加载成功 - 设备: {self.device}")
        except Exception as e:
            logger.error(f"OpenVINO 模型加载失败: {e}")
            logger.error("请确保模型已成功转换为OpenVINO格式，并检查OpenVINO环境是否正确安装。")
            raise
    # <<< 变化结束 >>>
    
    # ... (get_next_folder, cleanup_audio_dir_safe, _move_to_error_dir, 等方法保持不变) ...
    # ... (一直到 transcribe_audio_file)
    def get_next_folder(self):
        """
        获取下一个待处理的子文件夹（按照优先级）
        优先从audio_not中获取，如果audio_not没有，再从audio1获取
        返回：子文件夹的Path对象，如果没有则返回None
        """
        # 1. 首先检查高优先级的audio_not目录
        not_folders = []
        for item in self.audio_not_dir.iterdir():
            if item.is_dir():
                # 检查文件夹中是否有音频文件
                has_audio = any(item.rglob("*.opus")) or any(item.rglob("*.mp3")) or any(item.rglob("*.wav"))
                if has_audio:
                    not_folders.append(item)
        
        if not_folders:
            # 按名称排序，确保处理顺序一致
            # 分离数字文件夹和非数字文件夹
            numeric_folders = []
            non_numeric_folders = []  # 初始化非数字文件夹列表
            
            for folder in not_folders:
                if folder.name.isdigit():
                    numeric_folders.append(folder)
                else:
                    non_numeric_folders.append(folder)
            
            # 数字文件夹按数值排序，非数字文件夹按字符串排序
            numeric_folders.sort(key=lambda x: int(x.name))
            non_numeric_folders.sort(key=lambda x: x.name)
            
            # 优先返回数字文件夹，数字处理完再处理非数字
            next_folder = numeric_folders[0] if numeric_folders else (non_numeric_folders[0] if non_numeric_folders else None)
            
            if next_folder:
                logger.info(f"从高优先级池找到待处理子文件夹: {next_folder.name}")
                return next_folder, "not"  # 返回文件夹和来源标识
            else:
                logger.warning("高优先级池中找到文件夹但无法确定下一个（可能为空）")
                return None, None
        
        # 2. 如果audio_not没有，检查低优先级的audio1目录
        audio1_folders = []
        for item in self.audio1_dir.iterdir():
            if item.is_dir():
                # 检查文件夹中是否有音频文件
                has_audio = any(item.rglob("*.opus")) or any(item.rglob("*.mp3")) or any(item.rglob("*.wav"))
                if has_audio:
                    audio1_folders.append(item)
        
        if audio1_folders:
            # 按名称排序，确保处理顺序一致
            # 分离数字文件夹和非数字文件夹
            numeric_folders = []
            non_numeric_folders = []
            
            for folder in audio1_folders:
                if folder.name.isdigit():
                    numeric_folders.append(folder)
                else:
                    non_numeric_folders.append(folder)
            
            # 数字文件夹按数值排序，非数字文件夹按字符串排序
            numeric_folders.sort(key=lambda x: int(x.name))
            non_numeric_folders.sort(key=lambda x: x.name)
            
            # 优先返回数字文件夹，数字处理完再处理非数字
            next_folder = numeric_folders[0] if numeric_folders else (non_numeric_folders[0] if non_numeric_folders else None)
            
            if next_folder:
                logger.info(f"从低优先级池找到待处理子文件夹: {next_folder.name}")
                return next_folder, "audio1"  # 返回文件夹和来源标识
            else:
                logger.warning("低优先级池中找到文件夹但无法确定下一个（可能为空）")
                return None, None
        
        logger.info("audio_not和audio1中都没有可处理的子文件夹")
        return None, None
    
    def cleanup_audio_dir_safe(self):
        """
        安全地清理audio目录：
        1. 如果是文件夹，保持不变（等待处理）
        2. 如果是非文件夹，移动到错误目录
        """
        items_to_process = list(self.audio_dir.iterdir())
        
        if not items_to_process:
            return True, 0
        
        moved_count = 0
        folder_found = False
        
        for item in items_to_process:
            if item.is_dir():
                # 检查是否是有效的音频文件夹
                audio_count = len(list(item.rglob("*.opus"))) + len(list(item.rglob("*.mp3"))) + len(list(item.rglob("*.wav")))
                if audio_count > 0:
                    logger.info(f"发现有效音频文件夹: {item.name} ({audio_count}个音频文件)")
                    folder_found = True
                else:
                    logger.warning(f"文件夹 {item.name} 中没有音频文件，移动到错误目录")
                    self._move_to_error_dir(item)
                    moved_count += 1
            else:
                # 非文件夹，移动到错误目录
                logger.warning(f"发现非文件夹内容: {item.name}，移动到错误目录")
                self._move_to_error_dir(item)
                moved_count += 1
        
        if moved_count > 0:
            logger.info(f"已清理 {moved_count} 个非文件夹项目到错误目录")
        
        return folder_found, moved_count
    
    def _move_to_error_dir(self, item_path):
        """移动异常项目到错误目录"""
        try:
            # 直接使用原始名称，不添加时间戳
            target_path = self.error_dir / item_path.name
            
            # 如果目标已存在，先删除以允许覆盖
            if target_path.exists():
                if target_path.is_dir():
                    shutil.rmtree(str(target_path))
                else:
                    target_path.unlink()
                logger.warning(f"目标已存在，已删除旧文件: {target_path}")
            
            # 移动项目
            shutil.move(str(item_path), str(target_path))
            
            # 记录
            self.error_files.append(str(item_path))
            logger.info(f"已移动异常项目到错误目录: {item_path.name} -> {target_path.name}")
            
            return True
        except Exception as e:
            logger.error(f"移动异常项目失败 {item_path}: {e}")
            return False
    
    def move_folder_to_audio(self, folder_path, source_type="unknown"):
        """
        将文件夹从源目录移动到audio
        source_type: 'not' 或 'audio1'，表示文件夹来源
        """
        target_path = self.audio_dir / folder_path.name
        
        # 如果audio目录非空，先清理到错误目录
        if any(self.audio_dir.iterdir()):
            logger.warning(f"audio目录非空，正在清理到错误目录...")
            for item in self.audio_dir.iterdir():
                self._move_to_error_dir(item)
        
        # 移动文件夹
        try:
            shutil.move(str(folder_path), str(target_path))
            logger.info(f"✅ 已移动文件夹: {folder_path.name} -> {target_path} (来源: {source_type})")
            return target_path
        except Exception as e:
            logger.error(f"移动文件夹失败: {e}")
            return None
    
    def move_folder_to_audio_ok(self, folder_name):
        """
        将处理完成的文件夹从audio移动到audio_ok
        """
        source_path = self.audio_dir / folder_name
        target_path = self.audio_ok_dir / folder_name
        
        if not source_path.exists():
            logger.warning(f"源文件夹不存在: {source_path}")
            return False
        
        # 检查目标是否存在，如果存在则添加时间戳
        if target_path.exists():
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            new_name = f"{folder_name}_{timestamp}"
            target_path = self.audio_ok_dir / new_name
            logger.warning(f"目标文件夹已存在，重命名为: {new_name}")
        
        try:
            shutil.move(str(source_path), str(target_path))
            logger.info(f"✅ 已移动完成文件夹: {folder_name} -> {target_path}")
            return True
        except Exception as e:
            logger.error(f"移动完成文件夹失败: {e}")
            return False

    # <<< 变化开始: 重写整个转录逻辑 >>>
    # 创建一个简单的数据类来模拟 faster-whisper 的 segment 对象
    class Segment:
        def __init__(self, start, end, text):
            self.start = start
            self.end = end
            self.text = text

    def transcribe_audio_file(self, audio_file, output_base):
        """
        使用 OpenVINO 模型转录单个音频文件，先进行 VAD 分割
        """
        try:
            # 1. 使用 librosa 加载音频，自动重采样到16kHz
            audio_input, sampling_rate = librosa.load(audio_file, sr=16000)
            duration = len(audio_input) / sampling_rate
            
            # 2. 使用 VAD 获取语音时间戳
            # 注意：这里需要导入被注入后的 VAD 函数
            from faster_whisper.vad import get_speech_timestamps
            vad_options = VadOptionsCompat(
                threshold=0.5,
                min_speech_duration_ms=300,
                min_silence_duration_ms=100,
                speech_pad_ms=200
            )
            
            # 获取语音时间戳
            vad_timestamps = get_speech_timestamps(audio_input, vad_options, sampling_rate)
            
            if not vad_timestamps:
                logger.warning(f"未检测到语音: {audio_file}")
                # 如果没有检测到语音，处理整个音频
                segments = self._transcribe_full_audio(audio_input, sampling_rate)
            else:
                # 3. 对每个 VAD 片段进行转录
                all_segments = []
                for i, vad_seg in enumerate(vad_timestamps):
                    logger.info(f"处理 VAD 片段 {i+1}/{len(vad_timestamps)}")
                    
                    # 切割音频
                    start_sample = vad_seg['start']
                    end_sample = vad_seg['end']
                    segment_audio = audio_input[start_sample:end_sample]
                    
                    # 转录这个片段
                    segment_segments = self._transcribe_audio_segment(segment_audio, sampling_rate)
                    
                    # 调整时间戳（加上VAD片段的开始时间）
                    for seg in segment_segments:
                        seg.start += start_sample / sampling_rate
                        seg.end += start_sample / sampling_rate
                        all_segments.append(seg)
                
                segments = all_segments
            
            # 4. 写入字幕
            if self.enable_segment_merge:
                logger.info(f"正在执行智能片段合并... (原始片段数: {len(segments)})")
                segments = self.merge_segments(segments)
                logger.info(f"智能合并完成 (合并后片段数: {len(segments)})")
                
            self.write_subtitles(segments, output_base)
            
            return True, duration
            
        except Exception as e:
            logger.error(f"OpenVINO 转录失败 {audio_file}: {e}")
            import traceback
            traceback.print_exc()
            return False, 0
        finally:
            # 内存回收逻辑
            import gc
            gc.collect()
            
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                except:
                    pass
            logger.info(f"✓ 已清理内存和缓存")

    def _transcribe_full_audio(self, audio_input, sampling_rate):
        """转录整个音频（没有VAD分割）"""
        # 使用 processor 进行预处理
        input_features = self.processor(
            audio_input, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features
        
        # 使用 model.generate() 进行推理
        predicted_ids = self.model.generate(input_features, **self.generate_config)
        
        # 解码，并解析带时间戳的文本
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=False, decode_with_timestamps=True)
        
        # 将结果解析为与之前兼容的 segments 格式
        return self._parse_timestamps(transcription[0])

    def _transcribe_audio_segment(self, segment_audio, sampling_rate):
        """转录单个音频片段"""
        # 使用 processor 进行预处理
        input_features = self.processor(
            segment_audio, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        ).input_features
        
        # 使用 model.generate() 进行推理
        predicted_ids = self.model.generate(input_features, **self.generate_config)
        
        # 解码，并解析带时间戳的文本
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=False, decode_with_timestamps=True)
        
        # 将结果解析为与之前兼容的 segments 格式
        return self._parse_timestamps(transcription[0])


    def _parse_timestamps(self, transcription_with_ts):
        """
        解析 `batch_decode` 输出的带时间戳的文本
        格式: <|startofprev|> <|ja|> <|translate|> <|0.00|> Hello there.<|1.23|> <|1.23|> How are you?<|4.56|> ...
        """
        import re
        # 正则表达式匹配时间戳和紧随其后的文本
        # <|...|>
        timestamp_pattern = re.compile(r"<\|(\d+\.\d+)\|>(.*?)(?=<\||$)")
        
        matches = timestamp_pattern.findall(transcription_with_ts)
        
        segments = []
        # matches 结果是 [('0.00', ' Hello there.'), ('1.23', ' How are you?'), ...]
        for i in range(0, len(matches) -1):
            start_time_str, text = matches[i]
            end_time_str, _ = matches[i+1]
            
            start_time = float(start_time_str)
            end_time = float(end_time_str)
            text = text.strip()
            
            if text:
                segments.append(self.Segment(start=start_time, end=end_time, text=text))

        # 处理最后一个片段
        if len(matches) > 0:
            last_match = matches[-1]
            start_time = float(last_match[0])
            text = last_match[1].strip()
            if text:
                # 假设最后一个片段持续2秒，或者根据需要调整
                end_time = start_time + 2.0
                segments.append(self.Segment(start=start_time, end=end_time, text=text))

        return segments
    # <<< 变化结束 >>>
    
    def merge_segments(self, segments):
        """
        合并重复或包含的片段 (移植自 Faster-Whisper-TransWithAI-ChickenRice)
        """
        # 1. 首先按开始时间排序
        segments.sort(key=lambda s: s.start)
        merged = []
        i = 0
        while i < len(segments):
            # 2. 跳过空文本片段
            if not segments[i].text.strip():
                i += 1
                continue
            
            start, end, text = segments[i].start, segments[i].end, segments[i].text
            j = i + 1
            
            # 3. 向前合并：如果下一个片段的文本以当前片段文本开头（通常是 Whisper 的幻觉或重复修正）
            #    则采用下一个片段的结束时间和文本（因为它包含更完整的信息）
            while j < len(segments):
                if segments[j].text.startswith(text):
                    end, text = segments[j].end, segments[j].text
                    j += 1
                    continue
                break
                
            # 4. 向后检查：如果当前文本以某后续片段的文本结尾（包含关系）
            #    则延长当前片段的结束时间到该后续片段的结束时间
            k = j
            while k < len(segments):
                if not segments[k].text.strip():
                    break
                if text.endswith(segments[k].text):
                    end = segments[k].end
                    k += 1
                    continue
                break
                
            merged.append(self.Segment(start=start, end=end, text=text))
            i = j  # 移动索引到处理过的位置
        return merged

    def write_subtitles(self, segments, base_path):
        """
        写入字幕文件
        """
        from datetime import timedelta

        def fmt_srt(td):
            s = int(td.total_seconds())
            ms = int((td.total_seconds() - s) * 1000)
            return f"{s//3600:02}:{s%3600//60:02}:{s%60:02},{ms:03}"

        def fmt_vtt(td):
            s = int(td.total_seconds())
            ms = int((td.total_seconds() - s) * 1000)
            return f"{s//3600:02}:{s%3600//60:02}:{s%60:02}.{ms:03}"

        def fmt_lrc(td):
            s = int(td.total_seconds())
            cs = int((td.total_seconds() - s) * 100)
            return f"{s//60:02}:{s%60:02}.{cs:02}"

        # 确保输出目录存在
        Path(base_path).parent.mkdir(parents=True, exist_ok=True)
        
        # ---- SRT ----
        try:
            with open(f"{base_path}.srt", "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{fmt_srt(timedelta(seconds=seg.start))} --> {fmt_srt(timedelta(seconds=seg.end))}\n")
                    f.write(f"{seg.text}\n\n")
        except Exception as e:
            logger.error(f"写入SRT失败: {e}")
        
        # ---- VTT ----
        try:
            with open(f"{base_path}.vtt", "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for i, seg in enumerate(segments, 1):
                    f.write(f"{i}\n")
                    f.write(f"{fmt_vtt(timedelta(seconds=seg.start))} --> {fmt_vtt(timedelta(seconds=seg.end))}\n")
                    f.write(f"{seg.text}\n\n")
        except Exception as e:
            logger.error(f"写入VTT失败: {e}")
        
        # ---- LRC ----
        try:
            with open(f"{base_path}.lrc", "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"[{fmt_lrc(timedelta(seconds=seg.start))}]{seg.text}\n")
        except Exception as e:
            logger.error(f"写入LRC失败: {e}")

    # ... (process_current_folder, process_all_folders, 等方法保持不变) ...
    def process_current_folder(self):
        """
        处理当前audio文件夹中的所有音频文件
        """
        # 安全地检查audio目录
        has_folder, moved_count = self.cleanup_audio_dir_safe()
        
        if not has_folder:
            logger.warning("audio目录中没有有效的音频文件夹")
            return True  # 标记为处理完成（实际上没有需要处理的）
        
        # 获取当前文件夹（应该是唯一的文件夹）
        items = list(self.audio_dir.iterdir())
        current_folder = None
        
        for item in items:
            if item.is_dir():
                current_folder = item
                break
        
        if not current_folder:
            logger.warning("未找到有效文件夹")
            return True
        
        folder_name = current_folder.name
        logger.info(f"正在处理文件夹: {folder_name}")
        
        # 获取当前文件夹中的所有音频文件（递归查找）
        audio_extensions = ['.opus', '.mp3', '.wav', '.flac', '.m4a', '.aac']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(current_folder.rglob(f"*{ext}"))
        
        if not audio_files:
            logger.warning(f"文件夹 {folder_name} 中没有找到音频文件")
            # 移动空文件夹到错误目录
            self._move_to_error_dir(current_folder)
            return False
        
        logger.info(f"找到 {len(audio_files)} 个音频文件")
        
        # 统计信息
        success_count = 0
        fail_count = 0
        total_duration = 0
        
        # 处理每个音频文件
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"处理文件 ({i}/{len(audio_files)}): {audio_file.name}")
            
            # 计算输出路径：保持相对路径结构
            rel_path = audio_file.relative_to(current_folder)
            output_base = self.output_dir / folder_name / rel_path.with_suffix('')
            
            # 创建输出目录
            output_base.parent.mkdir(parents=True, exist_ok=True)
            
            # 转录音频
            success, duration = self.transcribe_audio_file(audio_file, str(output_base))
            
            if success:
                success_count += 1
                total_duration += duration
                logger.info(f"✓ 完成: {audio_file.name} (时长: {duration:.2f}s)")
            else:
                fail_count += 1
                logger.error(f"✗ 失败: {audio_file.name}")
        
        logger.info(f"文件夹处理完成: 成功 {success_count}, 失败 {fail_count}, 总时长: {total_duration:.2f}s")
        
        return fail_count == 0  # 如果没有失败，返回True
    
    def process_all_folders(self):
        """
        处理所有文件夹，连续处理直到所有文件夹都处理完
        """
        logger.info("\n" + "="*60)
        logger.info("开始连续处理所有文件夹（优先处理audio_not）")
        logger.info("="*60)
        
        cycle_count = 0
        has_more_folders = True
        
        while has_more_folders:
            cycle_count += 1
            logger.info(f"\n🚀 处理周期 #{cycle_count}")
            
            # 1. 检查audio目录是否为空，如果不为空就处理
            if any(self.audio_dir.iterdir()):
                logger.info("audio目录中有待处理文件夹")
                
                # 安全地清理audio目录
                has_folder, moved_count = self.cleanup_audio_dir_safe()
                
                if has_folder:
                    # 获取当前文件夹
                    items = list(self.audio_dir.iterdir())
                    current_folder = None
                    
                    for item in items:
                        if item.is_dir():
                            current_folder = item
                            break
                    
                    if current_folder:
                        folder_name = current_folder.name
                        logger.info(f"正在处理文件夹: {folder_name}")
                        
                        # 处理当前文件夹
                        success = self.process_current_folder()
                        
                        if success:
                            # 移动处理完成的文件夹到audio_ok
                            if self.move_folder_to_audio_ok(folder_name):
                                self.processed_folders.append(folder_name)
                                logger.info(f"✅ 文件夹 {folder_name} 处理完成并移动")
                            else:
                                self.failed_folders.append(folder_name)
                                logger.error(f"❌ 文件夹 {folder_name} 移动失败")
                        else:
                            self.failed_folders.append(folder_name)
                            logger.error(f"❌ 文件夹 {folder_name} 处理失败")
                    else:
                        logger.info("audio目录中没有有效文件夹")
                else:
                    logger.info("audio目录中没有有效音频文件夹")
            else:
                logger.info("audio目录为空")
            
            # 2. 按照优先级获取下一个文件夹
            next_folder, source_type = self.get_next_folder()
            if next_folder:
                # 移动文件夹到audio
                moved_folder = self.move_folder_to_audio(next_folder, source_type)
                if moved_folder:
                    logger.info(f"📂 已加载新文件夹: {moved_folder.name} (来自: {source_type})")
                    has_more_folders = True  # 继续处理下一个
                    
                    # 等待一下，避免过于频繁
                    logger.info("等待2秒后继续处理...")
                    time.sleep(2)
                else:
                    logger.error("移动新文件夹失败")
                    has_more_folders = False
            else:
                logger.info("🎉 所有文件夹处理完成！")
                has_more_folders = False
    
    def print_summary(self):
        """打印处理摘要"""
        logger.info("\n" + "="*60)
        logger.info("处理摘要")
        logger.info("="*60)
        logger.info(f"成功处理的文件夹: {len(self.processed_folders)}")
        if self.processed_folders:
            logger.info("  - " + "\n  - ".join(self.processed_folders))
        
        logger.info(f"处理失败的文件夹: {len(self.failed_folders)}")
        if self.failed_folders:
            logger.info("  - " + "\n  - ".join(self.failed_folders))
        
        logger.info(f"移动到错误目录的项目: {len(self.error_files)}")
        if self.error_files:
            logger.info("  - " + "\n  - ".join([f for f in self.error_files[:10]]))
            if len(self.error_files) > 10:
                logger.info(f"  ... 还有 {len(self.error_files) - 10} 个项目")
        
        # 统计剩余文件夹（按优先级）
        not_remaining = [f.name for f in self.audio_not_dir.iterdir() if f.is_dir()]
        audio1_remaining = [f.name for f in self.audio1_dir.iterdir() if f.is_dir()]
        
        logger.info(f"高优先级池(not)剩余文件夹: {len(not_remaining)}")
        if not_remaining:
            logger.info("  - " + "\n  - ".join(not_remaining))
        
        logger.info(f"低优先级池(audio1)剩余文件夹: {len(audio1_remaining)}")
        if audio1_remaining:
            logger.info("  - " + "\n  - ".join(audio1_remaining[:20]))
            if len(audio1_remaining) > 20:
                logger.info(f"  ... 还有 {len(audio1_remaining) - 20} 个文件夹")
        
        logger.info("="*60)
    
    def cleanup(self):
        """清理资源"""
        uninject_vad()
        logger.info("✓ 已清理 VAD")

    # ... (run, print_summary, cleanup等方法保持不变)
    def run(self, max_folders=None):
        """
        主运行函数
        max_folders: 最大处理文件夹数，None表示无限制
        """
        try:
            self.setup_vad()
            self.load_model()
            
            # 如果指定了最大处理文件夹数，修改逻辑
            if max_folders is not None:
                logger.info(f"最多处理 {max_folders} 个文件夹")
                
                processed_count = 0
                while processed_count < max_folders:
                    # 检查是否还有待处理文件夹
                    next_folder, source_type = self.get_next_folder()
                    if not next_folder:
                        logger.info("没有更多待处理文件夹")
                        break
                    
                    # 移动文件夹到audio
                    moved_folder = self.move_folder_to_audio(next_folder, source_type)
                    if not moved_folder:
                        logger.error("移动文件夹失败，停止处理")
                        break
                    
                    # 处理当前文件夹
                    folder_name = moved_folder.name
                    logger.info(f"正在处理文件夹: {folder_name} ({processed_count + 1}/{max_folders})")
                    
                    success = self.process_current_folder()
                    
                    if success:
                        # 移动处理完成的文件夹到audio_ok
                        if self.move_folder_to_audio_ok(folder_name):
                            self.processed_folders.append(folder_name)
                            logger.info(f"✅ 文件夹 {folder_name} 处理完成并移动")
                            processed_count += 1
                        else:
                            self.failed_folders.append(folder_name)
                            logger.error(f"❌ 文件夹 {folder_name} 移动失败")
                    else:
                        self.failed_folders.append(folder_name)
                        logger.error(f"❌ 文件夹 {folder_name} 处理失败")
                    
                    # 等待一下，避免过于频繁
                    if processed_count < max_folders:
                        logger.info("等待2秒后处理下一个文件夹...")
                        time.sleep(2)
                
                logger.info(f"已完成 {processed_count} 个文件夹（最大限制: {max_folders}）")
            else:
                # 连续处理所有文件夹
                self.process_all_folders()
            
            self.print_summary()
            
        except KeyboardInterrupt:
            logger.info("用户中断处理")
            self.print_summary()
        except Exception as e:
            logger.error(f"处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='连续文件夹音频转录处理器')
    parser.add_argument('--audio_not_dir', type=str,
                       default=str(PROJECT_ROOT / "audio_not"),
                       help='高优先级待处理文件夹池路径')
    parser.add_argument('--audio1_dir', type=str,
                       default=str(PROJECT_ROOT / "audio1"),
                       help='低优先级待处理文件夹池路径')
    parser.add_argument('--audio_dir', type=str,
                       default=str(PROJECT_ROOT / "audio"),
                       help='当前处理文件夹路径')
    parser.add_argument('--audio_ok_dir', type=str,
                       default=str(PROJECT_ROOT / "audio_ok"),
                       help='已完成文件夹路径')
    parser.add_argument('--output_dir', type=str,
                       default=str(PROJECT_ROOT / "sub"),
                       help='字幕输出目录')
    # <<< 变化：修改模型路径的默认值和帮助信息 >>>
    parser.add_argument('--model_path', type=str,
                       default=str(PROJECT_ROOT / "models" / "whisper-chickenrice-large-v2-ov"),
                       help='转换后的 OpenVINO Whisper 模型路径')
    # <<< 变化结束 >>>
    parser.add_argument('--max_folders', type=int, default=None,
                       help='最大处理文件夹数，None表示无限制')
    parser.add_argument('--list_only', action='store_true',
                       help='仅列出待处理文件夹，不实际处理')
    parser.add_argument('--skip_current', action='store_true',
                       help='跳过当前audio目录，直接从audio1拉取新文件夹')
    parser.add_argument('--use_batch', action='store_true',
                       help='使用批处理模式（需要更多VRAM）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批处理大小（仅在使用批处理模式时有效）')
    # <<< 变化：添加 segment_merge 参数 >>>
    parser.add_argument('--enable_segment_merge', action='store_true',
                       help='启用智能片段合并（默认：False）')
    # <<< 变化结束 >>>
    # <<< 变化：添加设备参数 >>>
    parser.add_argument('--device', type=str, default="GPU",
                       help='推理设备 (例如 GPU, CPU, AUTO)')
    # <<< 变化结束 >>>
    
    args = parser.parse_args()
    
    # 打印GPU信息
    print("=" * 60)
    print("Colab连续文件夹处理器 (OpenVINO GPU 加速版)")
    print("=" * 60)
    # ... (可以保留旧的打印信息，但现在主要关注OpenVINO)
    
    # <<< 变化：初始化 Processor >>>
    processor = ContinuousFolderProcessor(
        audio_not_dir=args.audio_not_dir,
        audio1_dir=args.audio1_dir,
        audio_dir=args.audio_dir,
        audio_ok_dir=args.audio_ok_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        device=args.device, # 使用命令行参数
        # compute_type 和 batch 参数在新的实现中不再直接使用，但保留以防万一
        use_batch=args.use_batch,
        batch_size=args.batch_size,
        enable_segment_merge=args.enable_segment_merge
    )
    # <<< 变化结束 >>>
    
    # ... (main函数剩余部分保持不变) ...
    if args.list_only:
        # 仅列出待处理文件夹
        print(f"待处理的子文件夹 (高优先级: {args.audio_not_dir}):")
        not_subfolders = []
        for item in Path(args.audio_not_dir).iterdir():
            if item.is_dir():
                audio_count = len(list(item.rglob("*.opus"))) + len(list(item.rglob("*.mp3"))) + len(list(item.rglob("*.wav")))
                not_subfolders.append((item.name, audio_count))
        
        if not_subfolders:
            for name, count in not_subfolders:
                print(f"  - {name} ({count}个音频文件)")
            print(f"  总计: {len(not_subfolders)}个文件夹")
        else:
            print("  (无)")
            
        print(f"\n待处理的子文件夹 (低优先级: {args.audio1_dir}):")
        subfolders = []
        for item in Path(args.audio1_dir).iterdir():
            if item.is_dir():
                audio_count = len(list(item.rglob("*.opus"))) + len(list(item.rglob("*.mp3"))) + len(list(item.rglob("*.wav")))
                subfolders.append((item.name, audio_count))
        
        if subfolders:
            for name, count in subfolders:
                print(f"  - {name} ({count}个音频文件)")
            print(f"  总计: {len(subfolders)}个文件夹")
        else:
            print("  (无)")
        
        # 检查当前audio目录
        print("\n当前audio目录内容:")
        audio_items = list(Path(args.audio_dir).iterdir())
        if audio_items:
            for item in audio_items:
                if item.is_dir():
                    print(f"  - 文件夹: {item.name}")
                else:
                    print(f"  - 文件: {item.name}")
        else:
            print("  (空)")
    else:
        # 如果跳过当前目录，先清理audio目录到错误目录
        if args.skip_current:
            logger.info("跳过当前目录，移动audio目录内容到错误目录...")
            for item in processor.audio_dir.iterdir():
                processor._move_to_error_dir(item)
        
        # 运行处理
        processor.run(max_folders=args.max_folders)

if __name__ == "__main__":
    main()

