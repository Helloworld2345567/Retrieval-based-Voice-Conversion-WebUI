# ADR-0002：使用浅兼容 adapter

- 状态：Accepted
- 日期：2026-10-31

## 背景

模块已迁入 `rvc.core`、`rvc.vc` 和 `rvc.separation`，但现有脚本和下游仍使用少量旧入口。复制实现会造成长期分叉，立即删除又会制造不必要的迁移中断。

## 决策

只为当前确有等价实现的路径保留浅转发：命令入口 `infer/cli.py`、`tools.pymss.cli`；根模块 `rvc.runtime`、`rvc.audio`、`rvc.convert_cli`、`rvc.cuda_graph`、`tools.cuda_graph`；以及主线推理代码仍使用且存在严格等价实现的 `infer.audio`、`infer.fcpe`、`infer.hubert`、`infer.rmvpe`、`infer.module` 推理子集、`infer.vc.pipeline/utils` 和固定五模型的 `tools.pymss` 导出。规范实现只存在于新模块。adapter 保留旧命令所需的 cwd/输出语义，并由契约测试覆盖。

不为已删除的 WebUI、训练、实时或完整 PyMSS interface 伪造 adapter。

## 后果

下游可分阶段迁移，轻量命令仍不导入推理引擎。新增兼容路径必须证明存在用户并注明移除条件；adapter 不得承载新业务逻辑，计划在后续有明确版本边界时弃用。
