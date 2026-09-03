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
- [Conv2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html)
  PyTorch 官方二维卷积 API。用于核对输入/输出通道、权重 shape、padding 和输出空间尺寸。
- [Training a Classifier](https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
  PyTorch 官方 CNN 分类教程。用于对照卷积、池化、展平、全连接、loss 和训练循环的数据流。
- [Adam](https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html)
  PyTorch 官方 Adam API。用于理解状态、默认学习率和优化器对照实验。
- [BatchNorm2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html)
  PyTorch 官方二维 BatchNorm API。用于理解 batch 统计量、运行统计量及 train/eval 行为。
- [Dropout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html)
  PyTorch 官方 Dropout API。用于理解训练时随机置零、缩放和评估时恒等行为。
- [RandomAffine](https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomAffine.html)
  Torchvision 官方仿射增强 API。用于设计并核对 MNIST 的小幅旋转和平移实验。

## Wisdom (Communities)

- 暂不添加。当前先以官方教程和仓库中的可运行实验为主。

## Gaps

- 后续需要补充 PyTorch 官方的可复现实验、验证集划分和模型保存资料。
- 若开始系统比较 CNN 错误样本，需要补充可靠的模型解释与校准资料。
