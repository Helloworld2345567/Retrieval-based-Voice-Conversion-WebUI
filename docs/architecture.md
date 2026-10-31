# 推理架构

本仓库是 CLI-only 的离线推理产品。支持的产品入口只有 RVC 变声和固定五模型的人声分离；WebUI、训练、实时/VST、PyMSS server、ensemble 与 workflow 不属于当前范围。范围决策见 [ADR-0001](adr/0001-cli-only-scope.md)。

## 目录与依赖方向

```text
rvc/
  cli.py, __main__.py        # 统一入口；只做命令分派
  core/                      # 路径、设备、外部可执行文件
  vc/                        # 变声 CLI、配置、音频和模型实现
  separation/                # 分离 CLI、typed request、worker、模型与 YAML
  runtime.py, audio.py, ...  # 一个迁移周期内的浅兼容 adapter
infer/cli.py                 # 旧变声命令 adapter
tools/pymss/                 # 旧分离命令 adapter
```

允许的依赖方向为：

```text
统一入口 / 兼容 adapter
          ↓
rvc.vc        rvc.separation
          ↘   ↙
           rvc.core
```

`rvc.core` 不导入领域模块；`rvc.vc` 与 `rvc.separation` 不相互导入。CLI parser、模型查询和顶层包导入保持轻量，不应初始化 Torch、DirectML 或下载模型。

## 领域边界

- `rvc.core.devices.DeviceConfig` 只表达设备、dtype 与半精度状态；硬件库在创建配置时延迟导入。
- `rvc.vc.config.VoiceConversionConfig` 增加变声专属的分段、HuBERT/RMVPE 路径和 CUDA Graph 策略。
- `rvc.separation.request.SeparationRequest` 是 CLI、服务与 DirectML 子进程之间的 typed request；worker seam 见 [ADR-0003](adr/0003-directml-worker-seam.md)。
- 两条推理链共享 FFmpeg/ffprobe 查找，但保留各自的解码、声道、重采样与输出策略。

## 公共 interface

稳定入口：

- 命令：`rvc`, `python -m rvc`, `rvc convert`, `rvc separate`。
- 共享基础：`rvc.core.DeviceConfig`、`select_device`、`find_executable` 和路径 helper。
- 变声：`rvc.vc.VoiceConversionConfig`、`VoiceConverter`、`AudioResult`（延迟导出）。
- 分离：`rvc.separation.request.SeparationRequest`、`rvc.separation.catalog.ModelSpec`、`rvc.separation.separator.MSSeparator`。

公共对象避免在 import 时探测硬件。领域内部的模型类和 helper 可用于测试或专用集成，但不承诺与上述 interface 相同的兼容级别。

## 资源合同

wheel 只携带运行代码、`rvc/separation/resources/*.yaml`、`LICENSE.pymss` 和 `LICENSE.pymss-core`。测试、`assets/` payload、日志、缓存和构建目录不得进入 wheel。

RVC 权重、索引、HuBERT、RMVPE 与 PyMSS 权重是外部可变运行数据，默认源码布局见 [`assets/README.md`](../assets/README.md)。源码检出模式使用项目内 `assets/` 与 `.codex/`；wheel 安装自动使用平台用户数据/缓存目录（Windows `%LOCALAPPDATA%/RVC`，Linux `XDG_DATA_HOME`/`XDG_CACHE_HOME`），也可用 `RVC_DATA_DIR`、`RVC_CACHE_DIR`、各资源环境变量或 CLI 参数覆盖。分离 YAML 只读并始终从已安装的 `rvc.separation` 包读取。

## 兼容矩阵

| interface | 状态 | 规范实现 / 说明 |
| --- | --- | --- |
| `rvc`, `python -m rvc` | 支持 | `rvc.cli:main` |
| `rvc.vc.*` | 支持 | 变声规范实现 |
| `rvc.separation.*` | 支持 | 五模型分离规范实现 |
| `infer/cli.py` | 迁移兼容 | 转发到 `rvc.vc.cli`；旧 cwd 语义保留 |
| `python -m tools.pymss.cli` | 迁移兼容 | 转发到 `rvc.separation.cli` |
| `rvc.runtime` | 迁移兼容 | 转发设备、路径与 `VoiceConversionConfig` |
| `rvc.audio`, `rvc.convert_cli`, `rvc.cuda_graph`, `tools.cuda_graph` | 迁移兼容 | 转发到 `rvc.vc` 中的规范模块 |
| `infer.audio/fcpe/hubert/rmvpe`, `infer.module` 推理子集、`infer.vc.pipeline/utils` | 迁移兼容 | 只转发仍有严格等价实现的推理对象；不恢复 `VC` 或训练类 |
| `tools.pymss` Python 导出 | 迁移兼容 | 只导出固定五模型、下载与音频 I/O；不恢复 server/workflow/ensemble |
| 旧 WebUI、训练、实时/VST | 不支持 | 不提供空壳或隐式恢复 |
| PyMSS server、workflow、ensemble 与其他模型 | 不支持 | 仅保留固定五模型 CLI |

兼容 adapter 必须保持浅层、可测试且不复制实现；生命周期决策见 [ADR-0002](adr/0002-compatibility-adapters.md)。

## 扩展规则

新增通用能力先证明同时服务两个领域，才进入 `rvc.core`。新增 CLI 功能必须保持帮助和查询命令的轻量导入。新增资源需明确属于不可变 package data 还是外部运行数据，并相应更新 packaging smoke。任何跨进程字段都先加入 `SeparationRequest`，不得重新引入无类型字典协议。
