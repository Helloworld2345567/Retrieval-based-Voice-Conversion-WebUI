# CLI 使用说明

安装与模型资源准备见 [README](../README.md)，模块边界与资源合同见 [架构说明](architecture.md)。安装项目后，console script 与模块入口等价：

```bash
python -m pip install -e .              # 源码开发
rvc --help
python -m rvc --help

python -m build --wheel                 # 发布 wheel
python -m pip install --no-deps dist/rvc_cli_inference-*.whl
```

先按 README 安装对应 CPU/DirectML/CUDA 依赖。wheel 不携带模型 payload；从仓库外运行时应使用显式路径、CLI 参数或环境变量指向可写的外部资源目录。帮助、分离模型列表和详情不会初始化推理设备或下载权重。

## 变声

```bash
python -m rvc convert --model MODEL.pth --input INPUT --output OUTPUT
python -m rvc convert --model MODEL.pth --list-speakers
```

`--model` 必填，先按给定路径查找，再尝试模型目录中的同名文件。`--list-speakers` 输出有效 ID 与名称；没有名称元数据时输出可用 ID 范围。除此之外，`--input`、`--output` 都必填。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--speaker-id` | 元数据最小 ID，否则 0 | 有元数据时仅允许其中的 ID；否则按 embedding 范围选择 |
| `--pitch` | 0 | 整数半音变调 |
| `--f0-method` | rmvpe | pm、rmvpe、fcpe；无 F0 模型不加载音高模型 |
| `--index` | 自动匹配 | 显式指定已有 added 索引 |
| `--index-rate` | 0.75 | 范围 0～1；0 禁用索引，其他值要求索引可用 |
| `--resample-sr` | 0 | 0 保持模型采样率，其他整数值至少 16000 |
| `--rms-mix-rate` | 1.0 | 音量包络混合，范围 0～1 |
| `--protect` | 0.33 | 辅音与呼吸保护，范围 0～0.5 |
| `--format` | 按输出扩展名，否则 wav | wav、flac、mp3、m4a；显式指定时同步调整输出扩展名 |
| `--recursive` | 关闭 | 递归目录输入，并保留相对子目录 |
| `--overwrite` | 关闭 | 允许替换已有输出文件 |
| `--device` | auto | auto、cpu、cuda、directml |
| `--device-id` | 自动 | 一个非负 CUDA 或 DirectML 设备编号 |

单文件输入的输出可指定文件或目录；无扩展名的输出按目录处理。目录输入要求输出目录，默认只扫描当前层、输出 WAV。目录扫描识别 `.wav/.flac/.mp3/.m4a/.ogg/.opus/.aac/.wma/.mp4/.mkv/.webm`，扩展名不区分大小写。输入内的输出子目录会从递归扫描中排除。覆盖输入文件或多个输入映射到同一输出时直接报错，即使指定 `--overwrite` 也不会执行。

自动索引匹配依据模型名、去掉 `_eN_sN` 后缀的实验名以及 `_spkid<ID>`。先优先匹配 speaker 和标准索引名，再按索引目录顺序及修改时间选择；默认目录顺序为 `assets/indices/`、`logs/`。显式路径中的 `trained` 文件名映射到对应 `added` 文件名。索引缺失、损坏、为空或特征维度不匹配时失败，不静默切换为无检索推理。

默认资源位于项目目录，可用环境变量覆盖：

| 环境变量 | 默认目录 |
| --- | --- |
| `weight_root` | `assets/weights/` |
| `outside_index_root` | `assets/indices/` |
| `index_root` | `logs/` |
| `rmvpe_root` | `assets/rmvpe/` |
| `hubert_root` | `assets/hubert_base/` |

新入口的相对输入、输出、显式模型/索引路径及相对环境覆盖值，以调用目录为基准。变声旧入口按项目根目录解释，详见下文。

## 人声分离

```bash
python -m rvc separate list
python -m rvc separate info vocals
python -m rvc separate download vocals
python -m rvc separate infer vocals --input INPUT --output OUTPUT_DIR
```

模型参数接受下表别名、中文任务名、完整文件名或去掉 `.ckpt` 的文件名，英文不区分大小写。

| 别名 / 任务 | 完整模型名 | 新入口输出后缀 |
| --- | --- | --- |
| `dereverb` / 去混响 | `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` | noreverb / reverb |
| `dereverb-aggressive` / 去混响（激进） | `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` | noreverb / reverb |
| `vocals` / 去伴奏 | `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | vocals / instrumental |
| `vocals-aggressive` / 去伴奏（激进） | `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | vocals / instrumental |
| `lead-vocal` / 提主旋律 | `model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | main_vocal / off_vocal |

### 模型查询与下载

- `list`：默认输出别名、任务名、架构；`--json` 输出详情列表。`--category` 对分类路径作不区分大小写的子串筛选，分类为 `reverb_echo_control/dereverb`、`vocal/vocal_extraction`、`karaoke`。兼容参数 `--all` 仍列出这五个模型。
- `info MODEL`：输出 JSON，包括模型与配置路径、架构、声部、输出后缀和预期权重大小；可指定 `--model-dir`。权重尚不存在时也可查询。
- `download MODEL`：下载一个模型；`download all` 下载五个模型，单个失败后继续其余任务。已有权重默认复用，`--force` 强制重新下载。`--supported-only` 为兼容参数，这五个模型均受支持。

`download` 与 `infer` 共用 `--model-dir`、`--source` 和 `--endpoint`。模型目录按以下优先级选择：`--model-dir` → `PYMSS_MODEL_DIR` → `weight_pymss_root` → 默认数据目录下的 `assets/pymss_weights/`。源码检出默认数据目录是项目根；wheel 安装使用平台用户数据目录，也可由 `RVC_DATA_DIR` 覆盖。相对显式目录以调用目录为基准。优先读取目录中平铺的权重，也兼容读取旧分类子目录中的已有同名权重。模型配置始终使用包内 `rvc/separation/resources/`。

`--source` 可选 `modelscope`（默认）、`huggingface`、`hf-mirror`。`--endpoint` 优先于该选项，按 `<endpoint>/<分类路径>/<完整模型名>` 获取文件。`infer` 缺权重时自动下载；兼容参数 `--download` 不会强制替换已存在的权重。

### 分离推理参数

`--input/-i` 必填，接受单个音频文件或目录。目录只扫描当前层；扩展名与变声相同，另支持 `.aiff/.aif`。输出为 `<输入文件名去扩展名>_<后缀>.<格式>`，采样率为 44100 Hz。

| 参数 | 默认值 | 行为 |
| --- | --- | --- |
| `--output/-o` | results | 输出目录 |
| `--save-as-folder` | 关闭 | 为每个输入建立同名结果子目录 |
| `--format` | wav | wav、flac、mp3、m4a |
| `--wav-bit-depth` | FLOAT | FLOAT、PCM_16、PCM_24 |
| `--flac-bit-depth` | PCM_16 | PCM_16、PCM_24 |
| `--mp3-bit-rate` | 320k | MP3 码率 |
| `--m4a-bit-rate` | 512k | M4A 码率 |
| `--m4a-codec` | aac | 使用当前 PyAV/FFmpeg 构建提供的编码器 |
| `--m4a-aac-at-quality` | 2 | 仅在编码器为 aac_at 时使用 |
| `--device` | auto | auto、cpu、cuda、directml |
| `--device-id` | 自动 | 非负设备编号；CUDA 可重复指定不同编号启用多卡 |
| `--tta` | 关闭 | 启用测试时增强，增加计算量 |
| `--debug` | 关闭 | 输出调试日志 |
| `--param KEY=VALUE` | 见下表 | 可重复指定；同名参数以最后一次为准 |

分离会覆盖已有同名输出文件；覆盖本次输入或多个输入映射到同一输出时会报错。

`--param` 仅接受以下参数，布尔值写为 `true/false`：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `batch_size` | CPU/CUDA 通常 1，vocals-aggressive 为 4；DirectML 为 1 | 每批分块数，正整数 |
| `chunk_size` | CPU/CUDA 352800；DirectML 88200 | 每块采样点数，正整数 |
| `overlap_size` | 见下文 | 重叠采样点数，非负且小于 chunk_size |
| `standardize` | false | 标准化输入，输出时恢复 |
| `normalize` | false | 两个输出声部共享峰值归一化增益 |
| `use_amp` | 选中 FP16 的 CUDA 设备为 true，其余 false | CUDA 自动混合精度；CPU/DirectML 不启用 CUDA AMP |
| `model_dtype` | auto | DirectML 的 auto、float16、float32；CPU/CUDA 的参数精度不由此选项控制 |
| `cuda_attention_backend` | default | default、auto、flash、cudnn、efficient、math、xformers；不可用时回退默认实现 |
| `stem_batch_size` | 0 | 0 同时处理所有声部，正整数分组处理；normalize=true 时仍合并处理以共享增益 |

CPU/CUDA 的默认 `overlap_size`：dereverb、dereverb-aggressive、vocals-aggressive 为 176400；vocals、lead-vocal 为 264600。DirectML 默认为 22050。

DirectML 参数精度优先取 `--param model_dtype=...`，其次为 `PYMSS_DML_MODEL_DTYPE`，最后为 `auto`。auto 先尝试 FP16；模型加载或尚未成功处理任何文件时出现可重试的计算错误，会在新进程中重试 FP32。内存不足不重试；显式选择 float16/float32 只执行该精度。Ctrl+C 会结束任务并回收工作进程。

## 设备与旧命令兼容

两条推理链的 `auto` 都先选择符合运行时资格规则的 CUDA 设备，再尝试 DirectML，最后使用 CPU。CUDA 自动选择按计算能力、显存排序；指定 `--device-id` 时限制对应编号。显式 `--device cuda/directml` 不可用时会报错。DirectML 需要 Windows 的对应依赖环境。

规范模块为 `rvc.core`（共享设备、路径、FFmpeg 查找）、`rvc.vc`（变声）和 `rvc.separation`（分离）。以下旧入口只作为迁移 adapter，不应成为新集成的依赖：

| 旧入口 | 规范入口 | 状态 |
| --- | --- | --- |
| `infer/cli.py` | `rvc.vc.cli` / `rvc convert` | 命令兼容 |
| `tools.pymss.cli` | `rvc.separation.cli` / `rvc separate` | 命令兼容 |
| `rvc.runtime` | `rvc.core.devices`, `rvc.core.paths`, `rvc.vc.config` | 浅转发 |
| `rvc.audio`, `rvc.convert_cli`, `rvc.cuda_graph`, `tools.cuda_graph` | 对应 `rvc.vc.*` 模块 | 浅转发 |
| `infer.audio/fcpe/hubert/rmvpe`、`infer.module` 推理子集、`infer.vc.pipeline/utils` | 对应 `rvc.vc.*` 对象 | 只保留严格等价的推理导出 |
| `tools.pymss` | `rvc.separation` 固定五模型 interface | 不含 server/workflow/ensemble |

旧 WebUI、训练、实时/VST、PyMSS server、workflow、ensemble 和固定五模型以外的分离后端不受支持。完整兼容矩阵见 [架构说明](architecture.md)。

```bash
# 源码检出
python infer/cli.py --model MODEL.pth --input INPUT --output OUTPUT
# wheel 安装后等价入口
python -m infer.cli --model MODEL.pth --input INPUT --output OUTPUT
python -m tools.pymss.cli infer vocals --input INPUT --output OUTPUT_DIR
python -m tools.pymss.cli list
```

变声旧入口将相对路径及相对资源环境变量解释为项目根目录；分离旧入口保持调用目录语义。两者调用同一推理核心。分离兼容子命令为 `infer/list/info/download`；旧 `list` 默认展示完整模型文件名，编码默认值与新入口相同。

旧分离命令的激进去伴奏后缀为 `vocals/other`，提主旋律为 `karaoke/other`；其余模型与新入口一致。

## 返回码

| 返回码 | 含义 |
| --- | --- |
| 0 | 命令成功，推理任务全部完成 |
| 1 | 模型、输入、推理或输出失败，或批处理中部分失败 |
| 2 | 命令行语法、选项或参数取值错误 |
| 130 | 用户中断 |

任务准备失败时直接退出。批处理中的逐文件解码、推理或保存失败会记录错误，并继续其余文件，最后返回汇总结果。空输入、缺失声部、NaN/Inf 和保存失败均按失败处理；错误模型名、缺失模型、非法 speaker 或不可用索引属于运行失败，返回 1。
