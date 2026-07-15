# Notes

- 使用中文教学。
- 用户已经完成纯 NumPy 的 MNIST MLP，不要从零重复讲解已经实现过的整体流程。
- 当前重点是理解 NumPy 实现如何逐步映射到 PyTorch，而不是立即重构成工业项目。
- `torch/torch_from_numpy.py` 是教学过渡版本，应保留“手动数据流程 + PyTorch 模型训练”的对照价值。
- `torch/train.py` 是下一阶段重点，用于学习 `torchvision.datasets.MNIST`、`Dataset` 和 `DataLoader`。
- 解释时持续标注 tensor shape，并围绕 forward、loss、backward、参数更新这条主线。
- 除非用户明确要求，否则不要直接替用户写完整答案；优先给一个小任务并等待反馈。
- 只有在用户通过解释、回答或练习证明掌握后，才新增 learning record。覆盖过的内容不等于已经掌握。
