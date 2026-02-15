import os
from pathlib import Path
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from transformers import WhisperProcessor
import openvino as ov # 导入 openvino 库

safe_temp_dir = "C:/AI_zimu_jihua/code/ChickenRice_v2/models/temp"
if not os.path.exists(safe_temp_dir):
    os.makedirs(safe_temp_dir)
# 设置环境变量
os.environ['TEMP'] = safe_temp_dir
os.environ['TMP'] = safe_temp_dir
print(f"!!! Forcing temporary directory to: {safe_temp_dir} to avoid encoding errors.")

# --- 配置 ---
# 1. 你的 PyTorch 模型文件夹路径
PYTORCH_MODEL_PATH = "C:/AI_zimu_jihua/code/ChickenRice_v2/models/whisper-large-v2-translate-zh-v0.2-st"

# 2. 中间 ONNX 模型的保存路径
ONNX_MODEL_PATH = "C:/AI_zimu_jihua/code/ChickenRice_v2/models/temp_onnx_model"

# 3. 最终 OpenVINO 模型的保存路径
OV_MODEL_PATH = "C:/AI_zimu_jihua/code/ChickenRice_v2/models/whisper-chickenrice-large-v2-ov"

# --- 步骤 1: 导出到 ONNX ---
print("="*60)
print(f"步骤 1: 开始将 PyTorch 模型导出到 ONNX...")
print(f"  - 输入模型: {PYTORCH_MODEL_PATH}")
print(f"  - 输出 ONNX 路径: {ONNX_MODEL_PATH}")

try:
    # 使用 ORTModelForSpeechSeq2Seq 来进行 ONNX 的转换
    ort_model = ORTModelForSpeechSeq2Seq.from_pretrained(PYTORCH_MODEL_PATH, export=True)
    # 保存 ONNX 模型文件
    ort_model.save_pretrained(ONNX_MODEL_PATH)
    
    # 同时保存处理器配置
    processor = WhisperProcessor.from_pretrained(PYTORCH_MODEL_PATH)
    processor.save_pretrained(ONNX_MODEL_PATH) # 先和ONNX放一起

    print("步骤 1: ONNX 模型导出成功！")
    print("="*60)

except Exception as e:
    print(f"步骤 1: ONNX 模型导出失败！错误: {e}")
    # 如果第一步就失败，就不继续了
    exit(1)


# --- 步骤 2: 将 ONNX 转换为 OpenVINO IR ---
print("\n" + "="*60)
print(f"步骤 2: 开始将 ONNX 模型转换为 OpenVINO IR...")
print(f"  - 输入 ONNX 模型: {Path(ONNX_MODEL_PATH) / 'encoder_model.onnx'}")
print(f"  - 输出 OpenVINO 路径: {OV_MODEL_PATH}")

# 确保最终输出目录存在
os.makedirs(OV_MODEL_PATH, exist_ok=True)

try:
    # Whisper ONNX 模型由3部分组成: encoder, decoder, decoder_with_past
    # 我们需要分别转换它们
    onnx_models_to_convert = {
        "encoder_model.onnx": "openvino_encoder_model.xml",
        "decoder_model.onnx": "openvino_decoder_model.xml",
        "decoder_with_past_model.onnx": "openvino_decoder_with_past_model.xml"
    }

    core = ov.Core()

    for onnx_name, ov_name in onnx_models_to_convert.items():
        onnx_file = Path(ONNX_MODEL_PATH) / onnx_name
        ov_file = Path(OV_MODEL_PATH) / Path(ov_name).with_suffix(".xml")
        
        if not onnx_file.exists():
            print(f"警告: 找不到 {onnx_file}，跳过转换。")
            continue
            
        print(f"  - 正在转换: {onnx_name} -> {ov_name}")
        
        # 加载 ONNX 模型
        model = core.read_model(model=str(onnx_file))
        
        # 转换并保存为 OpenVINO IR 格式 (.xml 和 .bin)
        ov.save_model(model, output_model=str(ov_file))

    # 将配置文件也复制到最终目录
    print("  - 正在复制模型配置文件...")
    import shutil
    for filename in os.listdir(ONNX_MODEL_PATH):
        if filename.endswith(".json"):
            shutil.copy(Path(ONNX_MODEL_PATH) / filename, Path(OV_MODEL_PATH) / filename)
            
    # 特别地，把 processor 的文件也复制过去
    processor.save_pretrained(OV_MODEL_PATH)

    print("步骤 2: OpenVINO IR 转换成功！")
    print("="*60)

    print(f"\n🎉 转换全部完成！你的 OpenVINO 模型已保存在: {OV_MODEL_PATH}")

except Exception as e:
    print(f"步骤 2: ONNX 到 OpenVINO 的转换失败！错误: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

