# 推理验证

## 验证结果

2026-10-31 在 Windows、Python 3.12.10、Torch/Torchaudio 2.4.1 CPU 和 Intel UHD 770 DirectML 环境中，完整测试集 **94 项通过、0 项失败、0 项跳过**。依赖一致性检查通过。

| 验证项 | 结果 |
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
- 中文及空格路径、单文件与批量处理、递归转换、输出冲突、部分失败和返回码。
- WAV、FLAC、MP3、M4A 的实际合成音频编码与解码。
- 模拟下载的复用及失败清理；DirectML 新进程重试、中断清理和 RMVPE 设备编号传递的契约。

## 重跑

先按 [README](../README.md) 创建并激活一种推理环境，准备 FFmpeg、ffprobe 和 Git。随后在仓库根目录安装测试依赖：

```bash
python -m pip install -r requirements-dev.txt
```

CPU 测试及设备契约检查：

```bash
python -B -m pytest -q
```

默认跳过六项实际 DirectML 测试。使用 Windows CPU/DirectML 依赖方案且具有可用适配器时，在 PowerShell 中启用它们：

```powershell
$env:RVC_TEST_DIRECTML = '1'
python -B -m pytest -q
Remove-Item Env:RVC_TEST_DIRECTML
```

启用后，DirectML 初始化或推理失败会使测试失败，不会静默跳过。该设置只影响测试。

数值回归需要仓库历史包含上述基线对象。完整克隆无需额外准备；浅克隆可执行 `git fetch --unshallow origin` 补齐历史。源码压缩包用户应改用 Git 克隆运行这组回归。

测试输出与缓存集中在 `.codex/`，不纳入版本管理。

## Limitations / Known Issues

- 未使用真实 RVC、HuBERT、RMVPE 和五个完整分离权重；合成模型结果不代表真实音质或分离效果。
- CUDA 硬件推理和 Linux 运行未实测；保留对应代码与安装方案。
- 实际 DirectML 测试采用小模型；完整模型的显存、速度和 FP16→FP32 回退未以完整权重验证。
- DirectML 的 `weight_norm` 算子由运行库回退到 CPU；通过不代表所有算子均由 GPU 执行或性能已验收。
