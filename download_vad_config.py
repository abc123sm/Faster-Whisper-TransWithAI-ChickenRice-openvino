import os
from transformers import WhisperFeatureExtractor, WhisperConfig

def download_vad_config():
    output_dir = "models/whisper-base"
    model_name = "openai/whisper-base"
    
    print(f"正在下载 {model_name} 的配置文件到 {output_dir} ...")
    
    # 确保目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载特征提取器配置
    try:
        feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        feature_extractor.save_pretrained(output_dir)
        print(f"特征提取器配置已保存。")
    except Exception as e:
        print(f"下载特征提取器失败: {e}")

    # 下载模型配置 (config.json)
    try:
        config = WhisperConfig.from_pretrained(model_name)
        config.save_pretrained(output_dir)
        print(f"模型配置已保存。")
    except Exception as e:
        print(f"下载模型配置失败: {e}")
        
    print(f"所有文件已保存在: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    download_vad_config()
