# 已完成纯 NumPy 的 MNIST MLP

用户已经在 `non_torch/` 中完成了 IDX 数据读取、归一化与展平、one-hot、Linear、ReLU、softmax cross entropy、SGD、mini-batch 训练、测试集评估和权重可视化，测试准确率约为 97.5%。这意味着后续教学不应重新从零介绍完整 MLP，而应以这些手写实现为参照，重点建立它们与 PyTorch API 及自动微分流程之间的映射。

## Evidence

仓库中存在四个按顺序串联并完成训练流程的 notebook：`01_read_data.ipynb`、`02_linear_relu.ipynb`、`03_softmax_loss.ipynb` 和 `04_train.ipynb`。

## Implications

下一步适合学习 `torch/torch_from_numpy.py` 中模型、损失函数和优化器的替换关系，再进入 `torch/train.py` 的 `Dataset` 与 `DataLoader`。
