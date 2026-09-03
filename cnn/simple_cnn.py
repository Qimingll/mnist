"""不包含 BatchNorm、Dropout 和数据增强的基础 CNN 版本。"""
"""
输出尺寸公式
H_out = floor((H_in + 2P - D×(K-1) - 1) / S + 1)

BatchNorm 是“把每一层收到的数据整理得更好训练”
Dropout 是“故意给网络制造困难，防止它死记硬背”
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 无论从项目根目录还是 cnn/ 目录运行，都能定位到已有的 MNIST 数据。
DATA_DIR = Path(__file__).resolve().parents[1] / "datasets"
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.1


def create_data_loaders():
    # CNN 要保留图像的二维空间结构：单张图片 shape 为 (1, 28, 28)。
    # 因此这里不能像 MLP 版本那样使用 view(-1) 展平成 (784,)。
    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    return train_loader, test_loader


def create_model():
    return nn.Sequential(
        # 输入mnist灰度图，有一个通道；使用16个卷积核；卷积核大小为3x3，padding=1保证输出大小不变，每个卷积核输出一个feature map，28*28
        # 把同一个局部区域想象成同时接受16种检查,在同一个空间位置上，模型得到16个值，它们组成该位置的16维特征描述
        # padding 会在输入边（两端）缘添加虚拟像素，stride 表示卷积核每次移动多少格
        # dilation 膨胀率（空洞率），1代表连续，2代表每隔一个像素采样，但采样点数量不变
        # (B, 1, 28, 28) -> (B, 16, 28, 28)
        nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),        
        nn.ReLU(),

        # (B, 16, 28, 28) -> (B, 16, 14, 14)
        # 2*2里面选max
        nn.MaxPool2d(kernel_size=2),

        # 每一个输出通道都代表一个卷积核，这层的卷积核作用于上一层输出的16个通道，输出一个feature map
        # 这层卷积核是16*3*3
        # (B, 16, 14, 14) -> (B, 32, 14, 14)
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
        nn.ReLU(),

        # (B, 32, 14, 14) -> (B, 32, 7, 7)
        nn.MaxPool2d(kernel_size=2),

        # (B, 32, 7, 7) -> (B, 1568)
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def train(model, train_loader, criterion, optimizer):
    print("\n=== 开始训练 ===")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            # forward：输出是 (B, 10) 的 logits，不需要手动加 softmax。
            logits = model(images)
            loss = criterion(logits, labels)

            # backward + 参数更新：和之前的 PyTorch MLP 完全相同。
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        average_loss = epoch_loss / len(train_loader)
        train_accuracy = correct / total
        print(
            f"Epoch {epoch + 1:2d}/{EPOCHS} | "
            f"loss: {average_loss:.4f} | "
            f"train acc: {train_accuracy * 100:.2f}%"
        )


def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    # 临时关闭梯度计算，节省显存和计算资源。
    with torch.no_grad():
        for images, labels in test_loader:
            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    test_accuracy = correct / total
    print(f"\n测试集准确率: {test_accuracy:.4f} ({test_accuracy * 100:.2f}%)")


def visualize_first_layer_kernels(model):
    """可视化训练后第一层的 16 个 3×3 卷积核。"""
    # model[0] 是第一层 Conv2d，weight shape 为 (16, 1, 3, 3)。
    kernels = model[0].weight.detach().cpu()
    max_abs_value = kernels.abs().max().item()

    fig, axes = plt.subplots(4, 4, figsize=(8, 8), constrained_layout=True)
    for kernel_index, ax in enumerate(axes.flat):
        kernel = kernels[kernel_index, 0]
        image = ax.imshow(
            kernel,
            cmap="seismic",
            vmin=-max_abs_value,
            vmax=max_abs_value,
            interpolation="nearest",
        )

        # 3×3 很小，直接把每个训练后的权重写在格子中更容易观察。
        for row in range(3):
            for col in range(3):
                value = kernel[row, col].item()
                text_color = "white" if abs(value) > max_abs_value * 0.55 else "black"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )

        ax.set_title(f"Kernel {kernel_index}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75, label="Weight")
    fig.suptitle("First-layer 3x3 kernels after training", fontsize=14)


def visualize_first_layer_feature_maps(model, test_loader):
    """用同一张测试图片观察第一层卷积加 ReLU 后的 16 张特征图。"""
    model.eval()
    images, labels = next(iter(test_loader))
    image = images[:1]
    label = labels[0].item()

    with torch.no_grad():
        conv_output = model[0](image)           # (1, 16, 28, 28)，ReLU 前
        feature_maps = model[1](conv_output)    # (1, 16, 28, 28)，ReLU 后
        prediction = model(image).argmax(dim=1).item()

    # 先显示这 16 张特征图共同对应的原始数字。
    original_fig, original_ax = plt.subplots(figsize=(3, 3), constrained_layout=True)
    original_ax.imshow(image[0, 0].cpu(), cmap="gray")
    original_ax.set_title(f"Input image | label={label}, prediction={prediction}")
    original_ax.axis("off")

    feature_maps = feature_maps[0].cpu()
    max_activation = feature_maps.max().item()

    fig, axes = plt.subplots(4, 4, figsize=(9, 9), constrained_layout=True)
    for channel_index, ax in enumerate(axes.flat):
        feature_map = feature_maps[channel_index]
        image_plot = ax.imshow(
            feature_map,
            cmap="inferno",
            vmin=0,
            vmax=max_activation,
        )
        ax.set_title(
            f"Channel {channel_index}\nmax={feature_map.max().item():.2f}",
            fontsize=9,
        )
        ax.axis("off")

    fig.colorbar(
        image_plot,
        ax=axes.ravel().tolist(),
        shrink=0.75,
        label="Activation after ReLU",
    )
    fig.suptitle("Feature maps after first Conv + ReLU", fontsize=14)


def main():
    train_loader, test_loader = create_data_loaders()
    model = create_model()

    print(f"训练集: {len(train_loader.dataset)} 张, {len(train_loader)} 个 batch")
    print(f"测试集: {len(test_loader.dataset)} 张, {len(test_loader)} 个 batch")
    print("\n模型结构:")
    print(model)

    # 用一个 batch 检查 CNN 的输入和输出 shape。
    sample_images, _ = next(iter(train_loader))
    print(f"\n输入 shape: {sample_images.shape}")
    print(f"输出 shape: {model(sample_images).shape}")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"参数总数: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    train(model, train_loader, criterion, optimizer)
    evaluate(model, test_loader)

    # 训练后再观察卷积核和特征图；训练前的卷积核只是随机初始化。
    visualize_first_layer_kernels(model)
    visualize_first_layer_feature_maps(model, test_loader)
    plt.show()


if __name__ == "__main__":
    main()
