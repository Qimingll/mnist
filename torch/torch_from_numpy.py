import torch
import torch.nn as nn
import numpy as np
import struct
import matplotlib.pyplot as plt
from pathlib import Path

# 这是 NumPy -> PyTorch 的过渡版本：
# 保留手写版的数据读取和 batch 逻辑，只把模型训练切到 PyTorch。
data_dir = Path(__file__).resolve().parents[1] / "datasets" / "MNIST" / "raw"

# 数据读取（沿用 NumPy 版的 IDX 读取）
image_path = data_dir / "train-images-idx3-ubyte"
label_path = data_dir / "train-labels-idx1-ubyte"

with open(image_path, 'rb') as f:
    _, train_num, rows, cols = struct.unpack('>IIII', f.read(16))
    x_train = np.frombuffer(f.read(), dtype=np.uint8).reshape(train_num, rows * cols).astype(np.float32) / 255.0

with open(label_path, 'rb') as f:
    f.read(8)
    y_train = np.frombuffer(f.read(), dtype=np.uint8).copy()        # astype转换数据类型时就新建了一块内存，

# 测试集
test_image_path = data_dir / "t10k-images-idx3-ubyte"
test_label_path = data_dir / "t10k-labels-idx1-ubyte"

with open(test_image_path, 'rb') as f:
    _, test_num, _, _ = struct.unpack('>IIII', f.read(16))
    x_test = np.frombuffer(f.read(), dtype=np.uint8).reshape(test_num, rows * cols).astype(np.float32) / 255.0

with open(test_label_path, 'rb') as f:
    f.read(8)
    y_test = np.frombuffer(f.read(), dtype=np.uint8).copy()

# NumPy → Tensor
x_train = torch.from_numpy(x_train)          # (60000, 784), float32，基于原numpy数组构建tensor，原dtype，共享内存
y_train = torch.tensor(y_train, dtype=torch.long)  # (60000,), 整数标签，新建一个tensor并指定dtype，之前是uint8
x_test  = torch.from_numpy(x_test)           # (10000, 784)
y_test  = torch.tensor(y_test, dtype=torch.long)   # (10000,)

print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
print(f"x_test:  {x_test.shape}, y_test:  {y_test.shape}")


# 2. 模型定义
# 对照 NumPy 版：Linear(784,128) + ReLU + Linear(128,10)，网络层不用加上softmax和crossentropy
# model对象，类型是nn.Sequential，但 PyTorch 让它可以像函数一样被调用，model(x) 是让这个模型对输入 x 做一次前向传播。
model = nn.Sequential(
    nn.Linear(784, 128),    # 你的 class Linear(784, 128)
    nn.ReLU(),              # 你的 class ReLU
    nn.Linear(128, 10)      # 你的 class Linear(128, 10)
)

# 看看model结构
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"参数总数: {total_params:,}")

# 3. 损失函数 & 优化器
# 对照 NumPy 版：SoftmaxCrossEntropy + SGD(lr=0.1)
# 注意：CrossEntropyLoss 内置了 softmax，直接传 logits 和整数标签
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. 训练
epochs = 10
batch_size = 64

train_losses = []

print(f"\n=== 开始训练 ===")
for epoch in range(epochs):
    # 打乱数据
    idx = torch.randperm(len(x_train))

    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(x_train), batch_size):
        batch_x = x_train[idx[i:i + batch_size]]
        batch_y = y_train[idx[i:i + batch_size]]

        # forward
        output = model(batch_x)           # z1→relu→z2
        loss = criterion(output, batch_y) # loss_fn.forward(z2, batch_y)

        # backward + update
        optimizer.zero_grad()  # 我的 backward 前不需要这步，PyTorch 需要
        loss.backward()        # 4行 backward 串起来
        optimizer.step()       # sgd.update(linear1) + sgd.update(linear2)

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1:2d}/{epochs} | loss: {avg_loss:.4f}")

# 5. 评估
model.eval()        # 把模型切换到评估模式，接下来这个模型不是在训练，而是在做验证/测试/推理。
with torch.no_grad():       # 暂时关闭自动求导（不记录梯度）。评估时不需要反向传播，因此不用保存计算图，可以减少内存占用、提高速度。
    # 训练集抽样准确率（前 1000 张）
    train_output = model(x_train[:1000])        # logits
    train_pred = train_output.argmax(dim=1)     # dim，就是把第几个参数压缩成1
    train_sample_acc = (train_pred == y_train[:1000]).float().mean().item()     # 最后要item()，因为前面都是tensor

    # 测试集准确率
    test_output = model(x_test)
    test_pred = test_output.argmax(dim=1)
    test_acc = (test_pred == y_test).float().mean().item()


print(f"\n=== 最终结果 ===")
print(f"训练集抽样准确率（前 1000）: {train_sample_acc:.4f}")
print(f"测试集准确率:             {test_acc:.4f} ({test_acc*100:.2f}%)")

# 6. 画 loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (PyTorch)')
plt.tight_layout()
plt.show()

# 7. 可视化第一层权重
"""
为什么两次神经元图像不一样，都是前16个
而且准确率不如手写的non_torch版本高
"""

fig, axes = plt.subplots(4, 4, figsize=(8, 8))
W1 = model[0].weight.detach().numpy()  # (128, 784)
for i, ax in enumerate(axes.flat):
    w = W1[i].reshape(28, 28)
    ax.imshow(w, cmap='seismic', vmin=-0.5, vmax=0.5)
    ax.axis('off')
plt.suptitle('First Layer Weights - 16 out of 128 neurons', fontsize=14)
plt.tight_layout()
plt.show()
