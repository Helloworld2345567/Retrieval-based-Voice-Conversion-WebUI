# RVC CLI 推理

离线语音音色转换和人声分离。支持 RVC v1/v2、有 F0/无 F0、单说话人/多说话人及 FAISS 检索；人声分离提供五个固定模型。运行设备为 CPU、NVIDIA CUDA 和 Windows DirectML。

## 安装

使用 **Python 3.12 x64**，以下命令均在仓库根目录执行。环境和缓存放在 `.codex/`，不纳入版本管理。先创建环境，再从三种硬件安装方案中选择一种。

Windows PowerShell（需先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)）：

```powershell
$env:UV_CACHE_DIR = "$PWD/.codex/uv-cache"
uv venv --python 3.12 --seed .codex/venv
. .codex/venv/Scripts/Activate.ps1
$env:PIP_CACHE_DIR = "$PWD/.codex/pip-cache"
```

Ubuntu 24.04：

```bash
sudo apt install python3.12 python3.12-venv ffmpeg libsndfile1
python3.12 -m venv .codex/venv
source .codex/venv/bin/activate
export PIP_CACHE_DIR="$PWD/.codex/pip-cache"
```

Windows 需提供 FFmpeg 和 ffprobe，可放入项目根目录或加入 `PATH`。三个 `requirments_*.txt` 文件名保持既有拼写。

### CPU / Windows DirectML

适用于 CPU，或 Windows 的 AMD/Intel 显卡；Linux 的 AMD/Intel 使用 CPU。

```bash
python -m pip install -r requirments_cpu_py312.txt
```

### NVIDIA CUDA 11.8

在新的虚拟环境中先安装对应 Torch，再安装其余依赖。

```bash
python -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirments_cu118_py312.txt
```

### NVIDIA CUDA 12.8

适用于 RTX 50 系等需要新版 CUDA 的显卡，同样使用独立虚拟环境。

```bash
python -m pip install torch==2.7.1+cu128 torchaudio==2.7.1+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirments_cu128_py312.txt
```

CPU/DirectML 使用 Torch 2.4.1；CUDA 使用 Torch 2.7.1。每个环境只安装一种方案。国内镜像不可用时，可将依赖文件中的索引替换为 PyPI 和对应的官方 PyTorch 索引，保留版本约束。

## 模型资源

```text
assets/
  hubert_base/        # config.json 与配套模型权重，包含 final_proj
  rmvpe/             # 按设备准备 rmvpe.pt 或 rmvpe.onnx
  weights/           # 用户的 RVC 推理 .pth
  indices/           # 用户的 added .index
  pymss_weights/     # 五个分离模型的权重
```

RVC 变声需要本地 `.pth` 推理模型和适配 RVC 的 Transformers 格式 HuBERT/ContentVec。后者包含 `final_proj`，可从 [VoiceConversionWebUI 模型仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 的 `hubert_base/` 获取；模型随附的 `preprocessor_config.json` 可以保留，但当前加载路径不要求该文件。

`.index` 由用户提供，`--index-rate 0` 时无需索引。有 F0 的模型选择 `--f0-method rmvpe` 时，CPU/CUDA 使用 `rmvpe.pt`，DirectML 使用同目录的 `rmvpe.onnx`；这两种权重也可从上述模型仓库获取。PM 无需额外音高模型，FCPE 使用 `torchfcpe` 附带的权重。无 F0 模型不加载音高模型。

分离配置随 `rvc/separation/resources/` 提供。`separate infer` 缺权重时下载选中的权重，已有权重直接复用；也可先执行 `separate download`。默认下载到 `assets/pymss_weights/`，通过 `--model-dir` 指定其他目录。

## 使用

在项目根目录运行：

```bash
python -m rvc --help
python -m rvc convert --help
python -m rvc separate infer --help

# 不使用检索索引
python -m rvc convert --model assets/weights/voice.pth --input input.wav --output output.wav --index-rate 0 --device cpu

# 选择多说话人模型的 speaker
python -m rvc convert --model assets/weights/voice.pth --list-speakers
python -m rvc convert --model assets/weights/voice.pth --speaker-id 1 --input input.wav --output output.wav --index assets/indices/voice_spkid1.index

# 目录递归转换，保留相对目录
python -m rvc convert --model assets/weights/voice.pth --input inputs --output outputs --recursive --format flac --index-rate 0

# 查询五个分离模型
python -m rvc separate list
python -m rvc separate infer model_bs_roformer_ep_368_sdr_12.9628.ckpt --input input.wav --output separated --device cpu
```

`--device` 可选 `auto`、`cpu`、`cuda`、`directml`。自动模式按现有显卡资格规则选择 CUDA，随后尝试 DirectML，最后 CPU。显式选择不可用设备会报错。使用 `--device-id` 选择设备。

输出格式支持 WAV、FLAC、MP3 和 M4A。详细参数、分离模型表、输出命名和旧入口约定见 [CLI 文档](docs/cli.md)。

## 代码与验证

生产实现位于 `rvc/`，模型网络与 CLI 分离；旧的 `infer/cli.py`、`tools/pymss/cli.py` 只转发命令。

激活已安装推理依赖的环境后运行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

测试临时文件和缓存位于 `.codex/`。结构回归使用小型随机模型及合成音频，需要 Git 历史中的固定基线提交，不下载真实权重。实际 DirectML 测试的启用方式、验证结果与覆盖范围见 [验证说明](docs/validation.md)。

## 许可证

项目使用 [MIT 许可证](LICENSE)。保留的第三方代码、模型来源和外部工具见 [许可与引用说明](MIT协议暨相关引用库协议)，分离模块附有 [PyMSS](rvc/separation/LICENSE.pymss) 和 [PyMSS Core](rvc/separation/LICENSE.pymss-core) 的许可原文。

## Limitations / Known Issues

- 本次重构验收限定为代码、CLI 和合成模型；不代表真实音质或五个完整分离权重的效果验收。
- 本次没有 NVIDIA 硬件，CUDA 实际推理未验证。
- DirectML 的合成测试与真实模型推理分别记录，不能互相替代。
