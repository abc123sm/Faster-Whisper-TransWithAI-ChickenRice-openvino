# 🎙️ Faster Whisper TransWithAI ChickenRice OpenVINO

[![GitHub Release](https://img.shields.io/github/v/release/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice)](https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice/releases)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

高性能音视频转录和翻译工具 - 基于 Faster Whisper 和 OpenVINO 深度优化，专为 Intel 硬件（CPU/GPU/NPU）打造的日文转中文版本。

High-performance audio/video transcription and translation tool - Optimized for Intel Hardware (OpenVINO) based on Faster Whisper.

## ⚠️ 重要声明 / Important Notice

> **本软件为开源软件 / This software is open source**
>
> 🔗 **开源地址 / Repository**: >https://github.com/abc123sm/Faster-Whisper-TransWithAI-ChickenRice-openvino
> 🔗 **原项目地址 / Original Repository**: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice

> 👥 **开发团队 / Development Team**: AI汉化组 (https://t.me/transWithAI)
>
> 本软件完全免费开源 / This software is completely free and open source

## 🙏 致谢 / Acknowledgments

本项目基于 `TransWithAI/Faster-Whisper-TransWithAI-ChickenRice` 进行 OpenVINO 移植和优化。

- 🚀 基于 [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) 开发
- 🐔 使用 [chickenrice0721/whisper-large-v2-translate-zh-v0.2-st](https://huggingface.co/chickenrice0721/whisper-large-v2-translate-zh-v0.2-st) 日文转中文优化模型
- 🧠 **OpenVINO 模型**: [boy1981chow/whisper-chickenrice-large-v2-openvino](https://huggingface.co/boy1981chow/whisper-chickenrice-large-v2-openvino)
- 🔊 使用 [TransWithAI/Whisper-Vad-EncDec-ASMR-onnx](https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx) 音声优化 VAD 模型


## ✨ 功能特性 / Features

- ⚡ **OpenVINO 加速**: 专为 Intel 平台优化，支持 Intel CPU (Core/Xeon)、集成显卡 (Iris Xe/UHD)、独立显卡 (Arc A系列) 及 NPU 加速。
- 🎯 **高精度日文转中文翻译**: 内置"海南鸡v2"日文转中文优化模型 (OpenVINO FP16量化版)。
- 📝 **多格式输出**: 支持SRT、VTT、LRC等多种字幕格式。
- 🎬 **音视频支持**: 支持常见音频(mp3/wav/flac等)和视频格式(mp4/mkv/avi等)。
- �️ **拖拽使用**: 支持将文件或文件夹直接拖拽到程序图标上运行。
- 🔧 **灵活配置**: 可自定义转录参数，满足不同场景需求。

## 📦 版本说明 / Package Variants

### 基础版 (Base Package)
- ✅ 包含所有运行依赖
- ✅ 音声优化 VAD 模型
- ❌ **不含** Whisper 模型（需自行下载并放置在 `models` 目录）

### 海南鸡版 (ChickenRice Edition)
- ✅ 包含所有运行依赖
- ✅ 音声优化 VAD 模型
- ✅ **内置 "海南鸡v2 5000小时" OpenVINO 模型**（开箱即用）

## 🚀 快速开始 / Quick Start

### 1. 系统要求 / System Requirements
- 操作系统: Windows 10 / 11 (64-bit)
- 硬件: 
  - Intel CPU (推荐 11代及以上)
  - Intel GPU (推荐 Iris Xe 或 Arc A系列独显)
  - 内存: 建议 16GB 及以上（如果是核显，就算是单线程也会占用8G共享显存）

### 2. 驱动更新 / Driver Update
为了获得最佳性能，请确保您的 Intel 显卡驱动已更新至最新版本。


### 3. 使用方法 / Usage

#### 方式一：拖拽运行 (推荐)
直接将音频文件、视频文件或包含媒体文件的**文件夹**拖拽到 `ChickenRice_v2.exe` 图标上即可开始处理。
- 程序会自动识别文件类型。
- 字幕文件将默认生成在源文件同级目录下。

#### 方式二：监控模式
直接双击运行 `ChickenRice_v2.exe`，程序将进入后台监控模式，自动扫描以下目录：
- `audio_not`: 高优先级处理目录
- `audio1`: 低优先级处理目录
处理完成后的文件会自动移动到 `audio_ok` 目录。

## 📖 详细文档 / Documentation

- 📝 [使用说明](使用说明.txt) - 详细的使用指南和参数配置
- ⚙️ [生成配置](generation_config.json5) - 转录参数配置文件

## 🛠️ 高级配置 / Advanced Configuration

### 命令行参数
您可以通过命令行运行程序以使用更多高级选项：

```bash
ChickenRice_v2.exe [输入路径] [选项]
```

常用选项:
- `--device [CPU|GPU|AUTO]`: 指定推理设备 (默认: GPU)
- `--output_dir [路径]`: 指定输出目录
- `--model_path [路径]`: 指定模型路径

### 转录参数调整
编辑 `generation_config.json5` 文件调整转录参数（如 `repetition_penalty`, `beam_size` 等）。

## 🔗 相关链接 / Links

- **海南鸡v2 5000小时 OpenVINO 模型**: https://huggingface.co/boy1981chow/whisper-chickenrice-large-v2-openvino
- **原项目**: https://github.com/TransWithAI/Faster-Whisper-TransWithAI-ChickenRice
- **音声优化 VAD 模型**: https://huggingface.co/TransWithAI/Whisper-Vad-EncDec-ASMR-onnx
- **OpenVINO**: https://docs.openvino.ai/

## 💡 常见问题 / FAQ

**Q: 运行速度慢？**
A: 请检查是否开启了 GPU 加速（默认开启）。如果是首次运行，OpenVINO 需要编译模型内核，可能需要几分钟时间，之后启动会很快，如果嫌这玩意占硬盘可以跑完就删掉，下次跑的时候重新搞。

**Q: 提示找不到 DLL？**
A: 请确保安装了最新的 VC++ 运行库，并且没有精简版的系统组件缺失。
ps. 可能还需要安装openvino runtime

**Q: 显存/内存不足？**
A: OpenVINO 会自动管理内存，但如果遇到内存不足，尝试关闭其他占用大量内存的程序，或在命令行中使用 CPU 模式运行。

**Q: intel独显是否可以使用？**
A: 我只有用13代的iris xe 96eu试过，我手上没有intel独显与其他任何型号的intel核显，不确定是否可以使用。

**Q: amd核显或独显是否可以使用？**
A: 你可以下载项目源码，装onnxruntime-directml，然后用dml跑，onnx模型跑dml速度很不错，至少肯定比原来那个项目纯用cpu强，不过最好的方式是你折腾一个rocm的版本，然后开源给大伙用。
ps. 我曾经用amd的rx580 4g跑过AI画图，用olive，比原版模型+dml速度快好几倍，而且4g显存也能跑1080x1080的图，效果与装了linux后用rocm跑的效果差不多（只是当时画图不能用lora，所以玩一下就丢了）。
ps2. 现在那张显卡坏了，我也不怎么玩游戏，所以才来折腾核显，如果有人愿意赞助个amd新显卡，我很乐意写个rocm的版本。

## 📞 技术支持 / Support

如遇到问题，请：
1. 查看[使用说明](使用说明.txt)
2. 检查显卡驱动是否为最新版本
3. 提交Issue到项目仓库

## ⭐ 小星星 / Star History

[![Star History Chart](https://api.star-history.com/svg?repos=abc123sm/Faster-Whisper-TransWithAI-ChickenRice-openvino&type=Date)](https://star-history.com/#abc123sm/Faster-Whisper-TransWithAI-ChickenRice-openvino&Date)

## 📄 许可证 / License

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

*本工具基于 Faster Whisper 和 OpenVINO 开发，旨在提供高性能的本地化字幕生成方案。*
