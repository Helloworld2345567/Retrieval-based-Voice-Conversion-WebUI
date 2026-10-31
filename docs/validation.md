# 推理验证

## 已记录验收快照

2026-10-31 的硬件验收在 Windows、Python 3.12.10、Torch/Torchaudio 2.4.1 CPU 和 Intel UHD 770 DirectML 环境中完成；当时的完整套件报告为 **94 passed、0 failed、0 skipped**，依赖一致性检查通过。这个数字是该提交和该硬件开关下的历史快照，不是永久固定的测试总数。

当前结果以 pytest 末行的 collected/passed/failed/skipped 为准：新增的结构与 packaging 合同会增加 collected 数量；默认未设置 `RVC_TEST_DIRECTML=1` 时，实际 DirectML 用例会显示为 skipped。文档不应把历史 94 项误作当前 CI 的硬编码门槛。

| 验证项 | 历史硬件验收结果 |
| --- | --- |
| RVC v1/v2 × F0/noF0 与原实现的 CPU 数值对比 | 四类模型最大绝对误差均为 0 |
| BS RoFormer / Mel Band RoFormer 与原实现的 CPU 数值对比 | 两类模型最大绝对误差均为 0 |
| 上述六类小模型的实际 DirectML 推理 | 全部通过；两个确定性的 RoFormer 另与 CPU 输出比较通过 |
| CLI、资源选择、音频编解码、错误处理、工作进程和分离文件输出 | 全部通过 |

数值基线是 Git 提交 `81eed5e8f68b6bed1789f682fe78cdd324495afc`。测试临时导出原模型源码，在隔离子进程中生成合成权重及输入，再将相同权重严格加载到新实现，并重置推理随机种子比较输出。无需保留旧生产目录或下载预训练权重。

## 测试覆盖

- checkpoint 配置、权重键名、说话人选择、索引匹配，以及损坏、空、错维度索引的失败行为。
- 无 F0 模型不加载音高模型；小型 RVC checkpoint 经完整 CLI/Pipeline 输出有效音频并释放模型，其中 HuBERT 特征使用确定性替身。
- 五个分离预设的固定配置与输出映射；两种小型分离 checkpoint 的装载、分块合成、双声部保存及重建。
- 新旧 CLI 帮助、模型列表、工作目录兼容、环境变量路径、参数错误和已删除命令；轻量命令不加载推理引擎。
- `rvc.core`、`rvc.vc`、`rvc.separation` 的模块边界，typed DirectML request 和兼容 adapter。
- 中文及空格路径、单文件与批量处理、递归转换、输出冲突、部分失败和返回码。
- WAV、FLAC、MP3、M4A 的实际合成音频编码与解码。
- 模拟下载的复用及失败清理；DirectML 新进程重试、中断清理和 RMVPE 设备编号传递的契约。
- `pyproject.toml`、console script 与 package data 合同；wheel 外部冒烟验证 YAML 和两个 PyMSS 许可文件可读，且测试、assets payload 和缓存未打包。

## 本地重跑

先按 [README](../README.md) 创建并激活一种推理环境，准备 FFmpeg、ffprobe 和 Git。随后在仓库根目录安装项目及测试依赖：

```bash
python -m pip install -e .
python -m pip install -r requirements-dev.txt
python -m pip check
python -B -m pytest -q
```

实际测试数以这次 pytest 输出为准。数值回归需要仓库历史包含上述基线对象；完整克隆无需额外准备，浅克隆可执行 `git fetch --unshallow origin`。源码压缩包用户应改用 Git 克隆运行这组回归。

默认跳过实际 DirectML 测试。使用 Windows CPU/DirectML 依赖方案且具有可用适配器时，在 PowerShell 中启用：

```powershell
$env:RVC_TEST_DIRECTML = '1'
python -B -m pytest -q
Remove-Item Env:RVC_TEST_DIRECTML
```

启用后，DirectML 初始化或推理失败会使测试失败，不会静默跳过。该设置只影响测试。

## packaging 与 wheel

需要 `build` 时执行：

```bash
python -m pip install build
python -m build --wheel
```

在新的临时虚拟环境中安装 wheel（硬件依赖已另行准备时使用 `--no-deps`），然后从仓库外运行：

```bash
python -m pip install --no-deps dist/rvc_cli_inference-*.whl
cd ..
rvc --help
rvc separate list
python -I -m infer.cli --help
python -I -m tools.pymss.cli list
python -I -c "import infer.audio, infer.module.models, infer.vc.pipeline, tools.cuda_graph, tools.pymss; from importlib.resources import files; p=files('rvc.separation'); assert p.joinpath('resources','model_bs_roformer_ep_368_sdr_12.9628.yaml').is_file(); assert p.joinpath('LICENSE.pymss').is_file(); assert p.joinpath('LICENSE.pymss-core').is_file()"
```

还应检查 wheel 文件列表不含 `tests/`、`assets/`、`.codex/`、`.zcode/` 或 `__pycache__`。测试输出与缓存集中在 `.codex/`，构建产物位于 `build/`、`dist/` 与 `*.egg-info/`；这些路径均不纳入版本管理。

GitHub Actions 使用完整 Git 历史，在 Windows Server 2022 与 Ubuntu 24.04 的 Python 3.12 CPU 环境运行 pytest、构建 wheel，并从 runner 临时目录执行上述 CLI/resource smoke。Linux 明确只安装官方 CPU Torch，不安装 DirectML；CUDA 和实际 DirectML 硬件验证不伪装成普通 CI 覆盖。

## Limitations / Known Issues

- 未使用真实 RVC、HuBERT、RMVPE 和五个完整分离权重；合成模型结果不代表真实音质或分离效果。
- 普通 CI 不覆盖 CUDA 硬件或实际 DirectML 适配器；对应能力需要专用硬件验收。
- 实际 DirectML 测试采用小模型；完整模型的显存、速度和 FP16→FP32 回退未以完整权重验证。
- DirectML 的 `weight_norm` 算子可由运行库回退到 CPU；通过不代表所有算子均由 GPU 执行或性能已验收。
