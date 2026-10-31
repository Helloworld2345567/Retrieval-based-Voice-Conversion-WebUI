# 本地模型资源

`assets/` 保存本机推理数据，不是 Python 包的一部分，也不会进入源码分发或 wheel。除本说明文件外，这些目录由用户创建或由 CLI 下载，并被 Git 忽略。

| 目录 | 内容 |
| --- | --- |
| `weights/` | RVC 推理 `.pth` 模型 |
| `indices/` | FAISS `added*.index` 检索索引 |
| `hubert_base/` | 带 `final_proj` 的 HuBERT/ContentVec 配置与权重 |
| `rmvpe/` | CPU/CUDA 的 `rmvpe.pt` 或 DirectML 的 `rmvpe.onnx` |
| `pymss_weights/` | 五个受支持分离模型的权重及下载暂存文件 |

默认路径相对于源码检出目录；安装 wheel 后，建议用 CLI 参数或环境变量指定可写的外部资源目录。可用覆盖项见 [CLI 文档](../docs/cli.md)。不要将模型权重、下载片段或生成结果提交到仓库。
