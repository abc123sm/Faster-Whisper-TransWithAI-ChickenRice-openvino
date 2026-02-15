# colab_batch_processor_auto_continuous.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/usr/local/cuda/"

import sys
try:
    import nvidia.cudnn
    cudnn_dir = os.path.dirname(nvidia.cudnn.__file__)
    cudnn_lib_dir = os.path.join(cudnn_dir, 'lib')
    # 将其添加到 LD_LIBRARY_PATH 的最前面
    os.environ['LD_LIBRARY_PATH'] = cudnn_lib_dir + ':' + os.environ.get('LD_LIBRARY_PATH', '')
    print(f"✅ 已成功注入 cuDNN 9 路径: {cudnn_lib_dir}")
except ImportError:
    print("⚠️ 未检测到 nvidia-cudnn-cu12，请确保已执行 pip install nvidia-cudnn-cu12")
except Exception as e:
    print(f"⚠️ 注入 cuDNN 路径时发生错误: {e}")

import logging
import shutil
import time
from pathlib import Path
import argparse
from faster_whisper import WhisperModel
import ctranslate2
import torch

# 项目路径
PROJECT_ROOT = "/content/drive/MyDrive/AI_zimu_jihua/faster_whisper_transwithai_chickenrice"
sys.path.insert(0, PROJECT_ROOT)

# 引入 VAD 注入模块
from src.faster_whisper_transwithai_chickenrice.injection import inject_vad, uninject_vad
from src.faster_whisper_transwithai_chickenrice.vad_manager import VadConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("v5.11版，排序按数字而非字符串")

def my_progress_callback(chunk_idx, total_chunks, device):
    """自定义的VAD进度回调函数"""
    progress_pct = (chunk_idx / total_chunks) * 100
    print(f"\r  [VAD] 正在处理音频块: {chunk_idx}/{total_chunks} ({progress_pct:.2f}%) on {device}", end="", flush=True)
    if chunk_idx == total_chunks:
        print()


class ColabBatchProcessor:
    # 主处理类：负责加载模型、注入 VAD、批量转录 opus
    def __init__(self, model_path, output_dir, device="cuda", compute_type="float16"):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = device
        self.compute_type = compute_type
        self.model = None

        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    def setup_vad(self):
        # 设置VAD注入
        logger.info("初始化 VAD ...")
        uninject_vad()
        logger.info("已清理历史VAD注入")
        cfg = VadConfig()
        cfg.onnx_model_path     = f"{PROJECT_ROOT}/models/whisper_vad.onnx"
        cfg.onnx_metadata_path  = f"{PROJECT_ROOT}/models/whisper_vad_metadata.json"
        cfg.threshold            = 0.5
        cfg.min_speech_duration_ms = 300
        cfg.min_silence_duration_ms = 100
        cfg.speech_pad_ms          = 200

        inject_vad("whisper_vad", cfg, progress_callback=my_progress_callback)
        logger.info("✓ VAD 注入完成")

    # ------------------------------------------------------------
    def load_model(self):
        # 加载Whisper模型
        logger.info(f"加载模型: {self.model_path}")

        try:
            self.model = WhisperModel(
                self.model_path,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info(f"✓ 模型加载成功 - 设备: {self.device}, 计算类型: {self.compute_type}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    # ------------------------------------------------------------
    def transcribe_audio(self, audio_file, base_out):
        # 转录单个音频，输出 SRT/VTT/LRC
        try:
            logger.info(f"正在处理文件 : {audio_file}")
            gen_cfg = dict(
                language="ja",
                task="translate",
                vad_filter=True,
                vad_parameters=dict(
                    threshold=0.5,
                    min_speech_duration_ms=300,
                    min_silence_duration_ms=100,
                    speech_pad_ms=200
                ),
                max_initial_timestamp=30,
                repetition_penalty=1.1
            )

            segments, info = self.model.transcribe(audio_file, **gen_cfg)
            segments = list(segments)  # 确保可迭代
            self.write_subtitles(segments, base_out)
            logger.info(f"✓ 完成: {audio_file}")
            del segments, info
            logger.info(f"✓ 已清理: segments info")
            success = True

        except Exception as e:
            logger.error(f"转录失败 {audio_file}: {e}")
            success = False
            
        finally:
            import gc
            gc.collect()
        
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            logger.info(f"✓ 已清理内存和缓存")
        
        return success

    # ------------------------------------------------------------
    def write_subtitles(self, segments, base):
        from datetime import timedelta
        
        disclaimer_text = "本字幕为AI字幕，未经人工校正润色，如有不准确的部分请谅解\n\nKameAI字幕计划\n使用模型 whisper： 海南鸡饭v2\nvad： Whisper-Vad-EncDec-ASMR-onnx"
        
        #判断字幕行数
        line_count = len(segments)
        write_disclaimer = line_count > 10
        
        if not segments:
            logger.info(f"无语音内容，跳过字幕生成: {base}")
            return

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
        
        # 声明字幕（前后各1秒）
        first_seg_start = 0.0
        first_seg_end = 1.0

        last_end = max(seg.end for seg in segments)
        last_seg_start = last_end
        last_seg_end = last_end + 1.0

        # ---- SRT ----
        with open(f"{base}.srt", "w", encoding="utf-8") as f:
            idx = 1

            # 头部声明
            if write_disclaimer:
                f.write(f"{idx}\n")
                f.write(f"{fmt_srt(timedelta(seconds=first_seg_start))} --> {fmt_srt(timedelta(seconds=first_seg_end))}\n")
                f.write(f"{disclaimer_text}\n\n")
                idx += 1

            for seg in segments:
                f.write(f"{idx}\n")
                f.write(f"{fmt_srt(timedelta(seconds=seg.start))} --> {fmt_srt(timedelta(seconds=seg.end))}\n")
                f.write(f"{seg.text}\n\n")
                idx += 1

            # 尾部声明
            if write_disclaimer:
                f.write(f"{idx}\n")
                f.write(f"{fmt_srt(timedelta(seconds=last_seg_start))} --> {fmt_srt(timedelta(seconds=last_seg_end))}\n")
                f.write(f"{disclaimer_text}\n\n")


        # ---- VTT ----
        with open(f"{base}.vtt", "w", encoding="utf-8") as f:
            f.write("WEBVTT\n\n")
            idx = 1

            if write_disclaimer:
                f.write(f"{idx}\n")
                f.write(f"{fmt_vtt(timedelta(seconds=first_seg_start))} --> {fmt_vtt(timedelta(seconds=first_seg_end))}\n")
                f.write(f"{disclaimer_text}\n\n")
                idx += 1

            for seg in segments:
                f.write(f"{idx}\n")
                f.write(f"{fmt_vtt(timedelta(seconds=seg.start))} --> {fmt_vtt(timedelta(seconds=seg.end))}\n")
                f.write(f"{seg.text}\n\n")
                idx += 1

            if write_disclaimer:
                f.write(f"{idx}\n")
                f.write(f"{fmt_vtt(timedelta(seconds=last_seg_start))} --> {fmt_vtt(timedelta(seconds=last_seg_end))}\n")
                f.write(f"{disclaimer_text}\n\n")


        # ---- LRC ----
        with open(f"{base}.lrc", "w", encoding="utf-8") as f:
            # 头部声明
            if write_disclaimer:
                f.write(f"[{fmt_lrc(timedelta(seconds=first_seg_start))}]{disclaimer_text}\n")

            for seg in segments:
                f.write(f"[{fmt_lrc(timedelta(seconds=seg.start))}]{seg.text}\n")

            # 尾部声明
            if write_disclaimer:
                f.write(f"[{fmt_lrc(timedelta(seconds=last_seg_start))}]{disclaimer_text}\n")


    # ------------------------------------------------------------
    def process_directory(self, audio_dir, output_dir):
        # 扫描整个目录，批量处理 opus 文件（单线程顺序处理）
        audio_dir  = Path(audio_dir)
        output_dir = Path(output_dir)

        opus_files = list(audio_dir.rglob("*.opus"))
        logger.info(f"找到 {len(opus_files)} 个 opus 文件")

        ok, fail = 0, 0

        # 单线程顺序处理每个文件
        for opus in opus_files:
            rel = opus.relative_to(audio_dir)
            out_base = output_dir / rel.parent / opus.stem
            out_base.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                result = self.transcribe_audio(str(opus), str(out_base))
                if result:
                    ok += 1
                    logger.info(f"处理完成: 成功 {ok}, 失败 {fail}")
                else:
                    fail += 1
                    logger.info(f"处理完成: 成功 {ok}, 失败 {fail}")
            except Exception as e:
                logger.error(f"处理失败 {opus}: {e}")
                fail += 1

        logger.info(f"处理完成: 成功 {ok}, 失败 {fail}")

    # ------------------------------------------------------------
    def cleanup(self):
        uninject_vad()
        logger.info("✓ 已清理 VAD")


class ContinuousFolderProcessor:
    """
    连续文件夹处理器（增强版）：
    1. 高优先级：先从audio_not文件夹拉取文件夹
    2. 低优先级：audio_not处理完后，从audio1文件夹拉取文件夹
    3. 检查audio文件夹是否有待处理文件
    4. 处理audio文件夹中的所有音频
    5. 处理完成后，将整个文件夹移动到audio_ok
    6. 继续处理下一个文件夹，直到所有文件夹处理完成
    """
    
    def __init__(self, 
                 audio_not_dir="/content/drive/MyDrive/AI_zimu_jihua/audio_not",  # 高优先级
                 audio1_dir="/content/drive/MyDrive/AI_zimu_jihua/audio1",        # 低优先级
                 audio_dir="/content/drive/MyDrive/AI_zimu_jihua/audio",
                 audio_ok_dir="/content/drive/MyDrive/AI_zimu_jihua/audio_ok",
                 output_dir="/content/drive/MyDrive/AI_zimu_jihua/sub",
                 model_path=None,
                 device="cuda",
                 compute_type="float16"):
        
        self.audio_not_dir = Path(audio_not_dir)  # 高优先级文件夹池
        self.audio1_dir = Path(audio1_dir)        # 低优先级文件夹池
        self.audio_dir = Path(audio_dir)          # 当前处理文件夹
        self.audio_ok_dir = Path(audio_ok_dir)    # 已完成文件夹
        self.output_dir = Path(output_dir)        # 字幕输出目录
        self.error_dir = Path(audio_dir).parent / "audio_error"  # 错误文件目录
        
        # 确保目录存在
        self.audio_not_dir.mkdir(parents=True, exist_ok=True)
        self.audio1_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.audio_ok_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.error_dir.mkdir(parents=True, exist_ok=True)
        
        # 模型参数
        if model_path is None:
            model_path = f"{PROJECT_ROOT}/models/whisper-chickenrice-large-v2"
        self.model_path = model_path
        self.device = device
        self.compute_type = compute_type
        
        # 状态跟踪
        self.processed_folders = []
        self.failed_folders = []
        self.error_files = []
        
        logger.info("=" * 60)
        logger.info("连续文件夹处理器初始化完成（支持高优先级audio_not）")
        logger.info(f"高优先级池: {self.audio_not_dir}")
        logger.info(f"低优先级池: {self.audio1_dir}")
        logger.info(f"当前处理: {self.audio_dir}")
        logger.info(f"完成目录: {self.audio_ok_dir}")
        logger.info(f"输出目录: {self.output_dir}")
        logger.info(f"错误目录: {self.error_dir}")
        logger.info("=" * 60)
    
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
            return next_folder, "not"  # 返回文件夹和来源标识
        
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
            return next_folder, "audio1"  # 返回文件夹和来源标识
        
        logger.info("audio_not和audio1中都没有可处理的子文件夹")
        return None, None
    
    def _move_to_error_dir(self, item_path):
        """移动异常项目到错误目录"""
        try:
            # 直接使用原名称，不添加时间戳
            target_path = self.error_dir / item_path.name
            
            # 移动项目
            shutil.move(str(item_path), str(target_path))
            
            # 记录
            self.error_files.append(str(item_path))
            logger.info(f"已移动异常项目到错误目录: {item_path.name}")
            
            return True
        except Exception as e:
            logger.error(f"移动异常项目失败 {item_path}: {e}")
            return False
    
    def move_folder_to_audio(self, folder_path, source_type):
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

    def zip_or_cleanup_sub_folder(self, sub_folder: Path):
        """
        如果字幕文件夹非空 -> 压缩成 zip 到 sub 目录
        如果为空 -> 删除
        """
        if not sub_folder.exists():
            return

        # 判断是否有字幕文件
        files = [p for p in sub_folder.rglob("*") if p.is_file()]
        if not files:
            shutil.rmtree(sub_folder)
            logger.info(f"🗑️ 字幕为空，已删除: {sub_folder.name}")
            return

        outsub_dir = sub_folder.parent.parent / "outsub"
        outsub_dir.mkdir(parents=True, exist_ok=True)
        zip_path = outsub_dir / sub_folder.name
        shutil.make_archive(
            base_name=str(zip_path),
            format="zip",
            root_dir=str(sub_folder)
        )

        shutil.rmtree(sub_folder)
        logger.info(f"📦 已打包字幕: {zip_path.name}.zip")




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
    
    def process_current_folder(self, processor):
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
        
        # 处理当前文件夹
        processor.process_directory(str(current_folder), str(self.output_dir / folder_name))
        
        return True
    
    def process_all_folders(self, processor):
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
                
                # 处理当前文件夹
                success = self.process_current_folder(processor)
                
                if success:
                    # 获取当前文件夹名称
                    items = list(self.audio_dir.iterdir())
                    current_folder = None
                    for item in items:
                        if item.is_dir():
                            current_folder = item
                            break
                    
                    if current_folder:
                        folder_name = current_folder.name
                        # 移动处理完成的文件夹到audio_ok
                        if self.move_folder_to_audio_ok(folder_name):
                            self.processed_folders.append(folder_name)
                            logger.info(f"✅ 文件夹 {folder_name} 处理完成并移动")
                            
                            sub_folder = self.output_dir / folder_name
                            self.zip_or_cleanup_sub_folder(sub_folder)
                        else:
                            self.failed_folders.append(folder_name)
                            logger.error(f"❌ 文件夹 {folder_name} 移动失败")
                else:
                    items = list(self.audio_dir.iterdir())
                    for item in items:
                        if item.is_dir():
                            self.failed_folders.append(item.name)
                            logger.error(f"❌ 文件夹 {item.name} 处理失败")
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


# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Colab批量音频转录处理器')
    parser.add_argument('--mode', type=str, choices=['single', 'continuous'], default='continuous',
                       help='处理模式: single=单文件夹, continuous=连续多文件夹')
    parser.add_argument('--audio_dir', type=str,
                       default="/content/drive/MyDrive/AI_zimu_jihua/audio",
                       help='音频目录路径（单文件夹模式）')
    parser.add_argument('--audio_not_dir', type=str,
                       default="/content/drive/MyDrive/AI_zimu_jihua/audio_not",
                       help='高优先级待处理文件夹池路径（连续模式）')
    parser.add_argument('--audio1_dir', type=str,
                       default="/content/drive/MyDrive/AI_zimu_jihua/audio1",
                       help='低优先级待处理文件夹池路径（连续模式）')
    parser.add_argument('--audio_ok_dir', type=str,
                       default="/content/drive/MyDrive/AI_zimu_jihua/audio_ok",
                       help='已完成文件夹路径（连续模式）')
    parser.add_argument('--output_dir', type=str,
                       default="/content/drive/MyDrive/AI_zimu_jihua/sub",
                       help='字幕输出目录')
    parser.add_argument('--model_path', type=str,
                       default=f"{PROJECT_ROOT}/models/whisper-chickenrice-large-v2",
                       help='Whisper模型路径')
    parser.add_argument('--max_folders', type=int, default=None,
                       help='最大处理文件夹数（连续模式），None表示无限制')
    
    args = parser.parse_args()
    
    # 打印GPU信息
    print("=" * 60)
    print("Colab批量音频转录处理器")
    print("=" * 60)
    
    try:
        if torch.cuda.is_available():
            print(f"GPU 设备: {torch.cuda.get_device_name(0)}")
            print(f"GPU 内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            print("CUDA 不可用")
    except:
        print("无法获取GPU信息")
    
    print("=" * 60 + "\n")
    
    processor = ColabBatchProcessor(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device="cuda",
        compute_type="float16"
    )
    
    if args.mode == 'single':
        # 单文件夹模式
        try:
            processor.setup_vad()
            processor.load_model()
            processor.process_directory(
                args.audio_dir,
                processor.output_dir
            )
        finally:
            processor.cleanup()
    else:
        # 连续多文件夹模式
        folder_processor = ContinuousFolderProcessor(
            audio_not_dir=args.audio_not_dir,
            audio1_dir=args.audio1_dir,
            audio_dir=args.audio_dir,
            audio_ok_dir=args.audio_ok_dir,
            output_dir=args.output_dir,
            model_path=args.model_path,
            device="cuda",
            compute_type="float16"
        )
        
        try:
            processor.setup_vad()
            processor.load_model()
            
            # 如果指定了最大处理文件夹数
            if args.max_folders is not None:
                logger.info(f"最多处理 {args.max_folders} 个文件夹")
                processed_count = 0
                
                while processed_count < args.max_folders:
                    # 检查是否还有待处理文件夹
                    next_folder, source_type = folder_processor.get_next_folder()
                    if not next_folder:
                        logger.info("没有更多待处理文件夹")
                        break
                    
                    # 移动文件夹到audio
                    moved_folder = folder_processor.move_folder_to_audio(next_folder, source_type)
                    if not moved_folder:
                        logger.error("移动文件夹失败，停止处理")
                        break
                    
                    # 处理当前文件夹
                    folder_name = moved_folder.name
                    logger.info(f"正在处理文件夹: {folder_name} ({processed_count + 1}/{args.max_folders})")
                    
                    success = folder_processor.process_current_folder(processor)
                    
                    if success:
                        # 移动处理完成的文件夹到audio_ok
                        if folder_processor.move_folder_to_audio_ok(folder_name):
                            folder_processor.processed_folders.append(folder_name)
                            logger.info(f"✅ 文件夹 {folder_name} 处理完成并移动")
                            
                            sub_folder = folder_processor.output_dir / folder_name
                            folder_processor.zip_or_cleanup_sub_folder(sub_folder)
                            processed_count += 1
                        else:
                            folder_processor.failed_folders.append(folder_name)
                            logger.error(f"❌ 文件夹 {folder_name} 移动失败")
                    else:
                        folder_processor.failed_folders.append(folder_name)
                        logger.error(f"❌ 文件夹 {folder_name} 处理失败")
                    
                    # 等待一下，避免过于频繁
                    if processed_count < args.max_folders:
                        logger.info("等待2秒后处理下一个文件夹...")
                        time.sleep(2)
                
                logger.info(f"已完成 {processed_count} 个文件夹（最大限制: {args.max_folders}）")
                folder_processor.print_summary()
            else:
                # 连续处理所有文件夹
                folder_processor.process_all_folders(processor)
                folder_processor.print_summary()
                
        except KeyboardInterrupt:
            logger.info("用户中断处理")
            folder_processor.print_summary()
        except Exception as e:
            logger.error(f"处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            processor.cleanup()


if __name__ == "__main__":
    main()