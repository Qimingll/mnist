# 已完成 PyTorch MLP 阶段

用户已经完成 `torch/torch_from_numpy.py` 的 NumPy 数据流程过渡版，以及 `torch/train.py` 的 `torchvision.datasets.MNIST`、`Dataset` 和 `DataLoader` 标准版。这意味着后续教学应把 PyTorch MLP 作为已知参照，当前重点转向 CNN 的局部连接、通道、特征图和受控实验。

## Evidence

仓库中的两个 PyTorch 脚本已经完成并单独提交；用户随后开始编写 `cnn/simple_cnn.py`，并主动追问卷积通道、padding、softmax 和实验可解释性。

## Implications

后续无需重新教授完整 PyTorch MLP 训练模板，应围绕现有 CNN 代码设计 shape 追踪、可视化和单变量对照实验。
