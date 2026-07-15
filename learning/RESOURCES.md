# PyTorch MNIST 学习资源

## Knowledge

- [PyTorch Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)
  PyTorch 官方基础教程总入口。用于按顺序学习 tensor、数据集、模型、autograd、优化和保存模型。
- [Datasets & DataLoaders](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)
  PyTorch 官方数据教程。用于理解 `Dataset` 表示什么，以及 `DataLoader` 如何完成 batching、shuffle 和迭代。
- [Build the Neural Network](https://docs.pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html)
  PyTorch 官方模型教程。用于理解 `nn.Module`、层的组织方式和 forward 数据流。
- [Automatic Differentiation with torch.autograd](https://docs.pytorch.org/tutorials/beginner/basics/autogradqs_tutorial.html)
  PyTorch 官方自动微分教程。用于把手写 backward 与 `loss.backward()` 对照起来。
- [Optimizing Model Parameters](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)
  PyTorch 官方训练循环教程。用于理解 loss、`zero_grad()`、`backward()` 和 `step()` 的协作关系。

## Wisdom (Communities)

- 暂不添加。当前先以官方教程和仓库中的可运行实验为主。

## Gaps

- 后续进入 CNN 阶段时，补充 PyTorch 官方卷积网络教程和卷积 shape 参考资料。
- 后续需要确定是否添加模型保存、验证集划分和可复现实验相关资料。
