# RVC CLI 推理

离线语音音色转换和人声分离。支持 RVC v1/v2、有 F0/无 F0、单说话人/多说话人及 FAISS 检索；人声分离提供五个固定模型。运行设备为 CPU、NVIDIA CUDA 和 Windows DirectML。

## 安装

使用 **Python 3.12 x64**（包元数据限制为 `>=3.12,<3.13`）。先创建环境，再选择一种硬件依赖方案。三个 `requirments_*.txt` 文件名保留既有拼写。

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

Windows 需提供 FFmpeg 和 ffprobe，可放入项目根目录或加入 `PATH`。

### CPU / Windows DirectML

适用于 CPU，或 Windows 的 AMD/Intel 显卡；Linux 的 AMD/Intel 使用 CPU。该文件在 Windows 安装 DirectML 依赖，在 Linux 通过 marker 跳过它们。

```bash
python -m pip install -r requirments_cpu_py312.txt
```

### NVIDIA CUDA 11.8

在新的虚拟环境中先从官方 PyTorch 索引安装对应 Torch，再安装其余依赖。

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

### 安装本项目

硬件依赖就绪后，开发或源码运行推荐 editable install：

```bash
python -m pip install -e .
rvc --help
```

`pyproject.toml` 从 `requirements-inference.txt` 读取通用运行依赖，并将 Torch/Torchaudio 限制在已支持的 2.4～2.7 系列、保留 Torch 2.4 所需的 `hyper-connections` 上限；具体 CPU/CUDA 构建和可选 DirectML 仍由上述硬件文件决定。已按硬件文件预装全部依赖时建议加 `--no-deps`，避免 pip 改变硬件栈。

构建和安装 wheel：

```bash
python -m pip install build
python -m build --wheel
python -m pip install --no-deps dist/rvc_cli_inference-*.whl
```

wheel 不包含模型、索引、测试或缓存。wheel 安装后的命令可在任意工作目录运行；外部模型目录需按下节和 [CLI 文档](docs/cli.md) 配置。

## 模型资源

```text
assets/
  hubert_base/        # config.json 与配套模型权重，包含 final_proj
  rmvpe/             # 按设备准备 rmvpe.pt 或 rmvpe.onnx
  weights/           # 用户的 RVC 推理 .pth
  indices/           # 用户的 added .index
  pymss_weights/     # 五个分离模型的权重
```

这些目录是本地运行数据，不会进入 Git 或 wheel，详见 [`assets/README.md`](assets/README.md)。源码检出时默认使用项目内目录；wheel 安装自动使用平台用户数据目录（Windows `%LOCALAPPDATA%/RVC`，Linux `XDG_DATA_HOME/rvc`），也可用 `RVC_DATA_DIR`、各资源环境变量、`--model-dir` 或显式路径覆盖。临时下载与 worker 文件使用对应用户缓存目录或 `RVC_CACHE_DIR`，不写入 `site-packages`。分离 YAML 和第三方许可文本随包安装。

RVC 变声需要本地 `.pth` 推理模型和适配 RVC 的 Transformers 格式 HuBERT/ContentVec。后者包含 `final_proj`，可从 [VoiceConversionWebUI 模型仓库](https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main) 的 `hubert_base/` 获取；模型随附的 `preprocessor_config.json` 可以保留，但当前加载路径不要求该文件。

`.index` 由用户提供，`--index-rate 0` 时无需索引。有 F0 的模型选择 `--f0-method rmvpe` 时，CPU/CUDA 使用 `rmvpe.pt`，DirectML 使用同目录的 `rmvpe.onnx`；这两种权重也可从上述模型仓库获取。PM 无需额外音高模型，FCPE 使用 `torchfcpe` 附带的权重。无 F0 模型不加载音高模型。

分离配置随 `rvc/separation/resources/` 提供。`separate infer` 缺权重时下载选中的权重，已有权重直接复用；也可先执行 `separate download`。源码检出默认下载到 `assets/pymss_weights/`；wheel 安装默认下载到用户数据目录，也可通过 `--model-dir` 或 `PYMSS_MODEL_DIR` 指定。

## 使用

`python -m rvc` 与安装后的 `rvc` console script 等价：

```bash
rvc --help
rvc convert --help
rvc separate infer --help

# 不使用检索索引
rvc convert --model assets/weights/voice.pth --input input.wav --output output.wav --index-rate 0 --device cpu

# 选择多说话人模型的 speaker
rvc convert --model assets/weights/voice.pth --list-speakers
rvc convert --model assets/weights/voice.pth --speaker-id 1 --input input.wav --output output.wav --index assets/indices/voice_spkid1.index

# 目录递归转换，保留相对目录
rvc convert --model assets/weights/voice.pth --input inputs --output outputs --recursive --format flac --index-rate 0

# 查询五个分离模型
rvc separate list
rvc separate infer model_bs_roformer_ep_368_sdr_12.9628.ckpt --input input.wav --output separated --device cpu
```

`--device` 可选 `auto`、`cpu`、`cuda`、`directml`。自动模式按现有显卡资格规则选择 CUDA，随后尝试 DirectML，最后 CPU。显式选择不可用设备会报错。使用 `--device-id` 选择设备。

输出格式支持 WAV、FLAC、MP3 和 M4A。详细参数、分离模型表、输出命名和旧入口约定见 [CLI 文档](docs/cli.md)。

## 代码、架构与验证

规范实现位于 `rvc/core/`、`rvc/vc/` 和 `rvc/separation/`；`infer/cli.py`、`tools/pymss/cli.py` 及少量 `rvc` 根模块是迁移兼容 adapter。依赖方向、公共 interface、资源合同和兼容矩阵见 [架构说明](docs/architecture.md)。

激活已安装推理依赖的环境后运行：

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

pytest 末行报告的是当前环境的 collected/passed/skipped 数量，不是固定产品版本号；硬件、平台 marker 和新增合同测试都会改变计数。结构回归使用小型随机模型及合成音频，需要 Git 历史中的固定基线提交，不下载真实权重。历史验收快照、当前重跑方式和实际 DirectML 开关见 [验证说明](docs/validation.md)。

## 许可证

项目使用 [MIT 许可证](LICENSE)。保留的第三方代码、模型来源和外部工具见 [许可与引用说明](MIT协议暨相关引用库协议)，分离模块附有 [PyMSS](rvc/separation/LICENSE.pymss) 和 [PyMSS Core](rvc/separation/LICENSE.pymss-core) 的许可原文。

## Limitations / Known Issues

- 代码和 CI 验收使用 CLI、资源合同及合成模型；不代表真实音质或五个完整分离权重的效果验收。
- 普通 CI 是 Windows/Linux CPU 基线，不验证 NVIDIA CUDA 或实际 DirectML 硬件。
- DirectML 的合成测试与真实模型推理分别记录，不能互相替代。
