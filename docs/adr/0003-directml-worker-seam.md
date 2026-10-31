# ADR-0003：DirectML 分离使用 worker seam

- 状态：Accepted
- 日期：2026-10-31

## 背景

DirectML 模型精度回退需要先以 FP16 尝试，再在可重试错误时用干净运行时执行 FP32。单进程重建无法可靠清除设备状态和显存，且中断时容易遗留资源。

## 决策

DirectML 分离由父进程生成 `SeparationRequest` JSON，启动 `rvc.separation.worker` 子进程执行一次精度尝试，并通过结果 JSON 和退出码返回。`auto` 精度的第二次尝试必须启动新进程。请求验证、序列化和精度覆盖集中在 typed request 中；父进程负责中断和 worker 回收。

CPU 与 CUDA 继续在当前进程运行，不经过该 seam。

## 后果

回退具有明确的进程隔离和可测试协议，代价是 DirectML 启动开销及临时请求文件。新增跨进程字段必须在 `SeparationRequest` 中显式定义并保持 JSON 往返；不得由环境或无类型字典暗中传递。
