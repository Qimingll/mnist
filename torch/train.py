import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# 1. 数据加载 — torchvision 标准写法

# 注意：datasets.MNIST 第一次运行会自动下载，但已经有了 IDX 文件，
# 设 download=False 直接用本地的
transform = transforms.Compose([
    transforms.ToTensor(),                    # uint8→float32，[0,255]→[0,1]
    transforms.Lambda(lambda x: x.view(-1))   # (channel, height, width)=(1, 28, 28) → (784,)
])

train_dataset = datasets.MNIST(
    root='./datasets', train=True, download=False, transform=transform)
test_dataset = datasets.MNIST(
    root='./datasets', train=False, download=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"训练集: {len(train_dataset)} 张, {len(train_loader)} 个 batch")
print(f"测试集: {len(test_dataset)} 张, {len(test_loader)} 个 batch")


# 2. 模型 — 和手写版一一对应
model = nn.Sequential(
    nn.Linear(784, 128),        # (in_features, out_features)，但内部形状不是这样，是倒过来的，128行（个）神经元
    nn.ReLU(),
    nn.Linear(128, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

total_params = sum(p.numel() for p in model.parameters())
print(f"参数总数: {total_params:,}")


# 3. 训练 — DataLoader 自动 shuffle + mini-batch
epochs = 10
train_losses = []

print("\n=== 开始训练 ===")
for epoch in range(epochs):
    epoch_loss = 0.0
    n_batches = 0

    for images, labels in train_loader:      # 不用手写 range(0,N,batch_size) 了
        output = model(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1:2d}/{epochs} | loss: {avg_loss:.4f}")


# 4. 评估 — 完整测试集
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        pred = output.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print(f"\n测试集准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")

# 5. 画 loss 曲线
plt.figure(figsize=(6, 4))
plt.plot(train_losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (PyTorch — DataLoader)')
plt.tight_layout()
plt.show()

# 6. 可视化第一层权重
fig, axes = plt.subplots(4, 4, figsize=(8, 8))
W1 = model[0].weight.detach().numpy()  # (128, 784)
for i, ax in enumerate(axes.flat):
    w = W1[i].reshape(28, 28)
    ax.imshow(w, cmap='seismic', vmin=-0.5, vmax=0.5)
    ax.axis('off')
plt.suptitle('First Layer Weights — 16 out of 128 neurons', fontsize=14)
plt.tight_layout()
plt.show()
