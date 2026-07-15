# Mission: 从 NumPy 手写神经网络过渡到 PyTorch

## Why

通过 MNIST 亲手走完从底层 NumPy 实现到标准 PyTorch 工作流的迁移，建立对数据流、前向传播、反向传播、损失计算和参数更新的可靠直觉。最终能够理解并独立编写常见的 PyTorch 训练代码，而不是只会复制框架模板。

## Success looks like

- 能解释 NumPy 手写组件与 PyTorch API 的一一对应关系
- 能追踪 MLP 中 `(batch, 784) -> (batch, 128) -> (batch, 10)` 的 shape 变化
- 能独立写出并解释 `Dataset`、`DataLoader`、模型、损失函数、优化器、训练和评估流程
- 能说明 `zero_grad()`、`backward()` 和 `step()` 分别承担什么职责
- 在理解 MLP 训练流程后，能够继续实现并解释 MNIST CNN

## Constraints

- 当前处于学习阶段，优先理解原理和流程，不追求过早工程化
- 每次只推进一个紧密范围内的知识点，并结合仓库中的实际代码
- 优先让学习者自己思考和编写；先给思路或小任务，卡住后再增加提示
- 运行 Python 时使用 conda 的 `mnist` 环境，并关注解释器是否正确

## Out of scope

- 暂不追求工业级训练框架、复杂配置系统和分布式训练
- 暂不引入 Transformer、生成模型等与当前 MNIST 路线无关的主题
- 在 MLP 与 PyTorch 基础未掌握前，不提前堆叠复杂 CNN 技巧
