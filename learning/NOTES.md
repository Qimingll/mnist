# Notes

- 使用中文教学。
- 所有 `learning/` HTML 课程和参考页统一使用深色 `course-theme` 风格；优先复用 `learning/assets/lesson.css`，不要重新创建浅色页面主题。
- 用户已经完成纯 NumPy 的 MNIST MLP，不要从零重复讲解已经实现过的整体流程。
- PyTorch MLP 阶段已经完成；当前重点是理解 CNN 的局部连接、输入/输出通道、卷积核、特征图和池化，而不是立即重构成工业项目。
- `torch/torch_from_numpy.py` 是教学过渡版本，应保留“手动数据流程 + PyTorch 模型训练”的对照价值。
- `torch/train.py` 已完成，作为 CNN 训练循环的参照；当前基础实现是 `cnn/simple_cnn.py`。
- 解释时持续标注 tensor shape，并围绕 forward、loss、backward、参数更新这条主线。
- MNIST CNN 已接近 99% 时，不以单次最终准确率的小幅变化判断组件优劣；优先比较多随机种子、收敛速度、训练/验证差距、鲁棒性和错误样本。
- CNN 进阶实验一次只改变 Adam、BatchNorm、Dropout 或数据增强中的一个变量。
- 当前需要巩固 CNN 权重共享：重点区分“MLP 的位置专属连接”和“同一卷积核在所有空间位置复用”；不要把平移等变误说成最终分类完全平移不变。
- 除非用户明确要求，否则不要直接替用户写完整答案；优先给一个小任务并等待反馈。
- 只有在用户通过解释、回答或练习证明掌握后，才新增 learning record。覆盖过的内容不等于已经掌握。
