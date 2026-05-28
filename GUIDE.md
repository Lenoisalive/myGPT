# 🚀 Bigram Language Model - 完整使用指南

## 📋 目录
1. [项目概述](#项目概述)
2. [快速开始](#快速开始)
3. [详细步骤](#详细步骤)
4. [训练过程](#训练过程)
5. [生成文本](#生成文本)
6. [可视化](#可视化)
7. [常见问题](#常见问题)

---

## 项目概述

这是一个从零开始实现的 **Bigram Language Model** (V1 版本)。

### 🎯 核心原理

```
P(next_token | current_token)
```

模型学习字符之间的转移概率：
- 输入: `"h"` → 预测: `"e"`
- 输入: `"e"` → 预测: `"l"`
- 输入: `"l"` → 预测: `"l"`

### ✨ 特点

- ✅ 最简单的语言模型实现
- ✅ 易于理解和调试
- ✅ 训练速度快
- ✅ 可以学会基本的字符组合规律
- ❌ 没有长程依赖理解
- ❌ 没有 Attention 机制

---

## 快速开始

### 1️⃣ 检查环境

```bash
# 确认虚拟环境已激活
which python  # 应该显示 .venv/bin/python

# 检查 PyTorch
python -c "import torch; print('PyTorch:', torch.__version__)"
```

### 2️⃣ 快速测试

```bash
python quick_test.py
```

应该看到：
```
✅ 测试 Tokenizer
✅ 测试 Bigram Model
✅ 测试 Dataset
```

### 3️⃣ 开始训练

```bash
python train.py
```

### 4️⃣ 生成文本

```bash
python generate.py
```

---

## 详细步骤

### 步骤 1: 准备数据

确保 `data/input.txt` 存在：

```bash
ls -lh data/input.txt
```

可以使用 `process_data.py` 查看数据统计：

```bash
python process_data.py
```

会显示：
- 📁 文件大小
- 📝 字符统计
- 🔤 词汇表信息
- 🔢 编码示例

### 步骤 2: 测试 Dataset

```bash
python dataset.py
```

验证：
- ✅ 数据加载正确
- ✅ Token 化成功
- ✅ x 和 y 正确错位一位

输出示例：
```
x[0][1] = 45  ==  y[0][0] = 45  ✓
x[0][2] = 23  ==  y[0][1] = 23  ✓
```

### 步骤 3: 测试模型

```bash
python model.py
```

检查：
- ✅ 模型创建成功
- ✅ 参数量正确
- ✅ 前向传播正常
- ✅ 生成功能正常

---

## 训练过程

### 启动训练

```bash
python train.py
```

### 训练输出

```
============================================================
  🚀 开始训练 Bigram Language Model
============================================================

📋 训练配置:
   设备: mps
   批次大小: 32
   上下文长度: 128
   最大迭代: 5000
   学习率: 0.0003
   评估间隔: 500

📊 数据统计:
   词汇表大小: 65
   训练集大小: 1,003,854 tokens
   验证集大小: 111,540 tokens

============================================================
  🔨 创建模型
============================================================
✅ 模型初始化完成
   词汇表大小: 65
   参数量: 4,225

============================================================
  🏋️  开始训练循环
============================================================

📈 步数     0/5000
   训练损失: 4.1732
   验证损失: 4.1698
   用时: 2.3秒
   💾 保存最佳模型 (验证损失: 4.1698)

📝 生成样本文本:
────────────────────────────────────────────────────────────
zKjxWBpQ
m3u!YtHv&...
────────────────────────────────────────────────────────────
.........

📈 步数   500/5000
   训练损失: 2.4823
   验证损失: 2.4956
   用时: 45.2秒
   💾 保存最佳模型 (验证损失: 2.4956)
...
```

### 训练期间

- 每 **500 步**评估一次
- 每 **1000 步**生成样本文本
- 自动保存最佳模型到 `checkpoints/best_model.pt`
- 记录训练日志到 `training_log.json`

### 预期结果

| 迭代次数 | 训练损失 | 验证损失 | 生成质量 |
|---------|---------|---------|---------|
| 0       | ~4.17   | ~4.17   | 完全随机 |
| 500     | ~2.50   | ~2.51   | 有一些字母组合 |
| 2000    | ~2.30   | ~2.32   | 能看到简单单词 |
| 5000    | ~2.20   | ~2.25   | 有意义的短语片段 |

### 停止训练

按 `Ctrl+C` 可以提前停止（模型会自动保存）。

---

## 生成文本

### 交互式模式（推荐）

```bash
python generate.py
```

或

```bash
python generate.py interactive
```

#### 使用方式

```
💬 提示 (温度=0.8): To be or not to be

⏳ 正在生成...
────────────────────────────────────────────────────────────
To be or not to be, that is the question...
────────────────────────────────────────────────────────────

💬 提示 (温度=0.8): temp 1.2
✅ 温度已设置为 1.2

💬 提示 (温度=1.2): Hello

💬 提示 (温度=0.8): quit
👋 再见!
```

#### 温度参数说明

- **0.5**: 更确定性，输出更保守
  ```
  The sun is shining in the sky.
  ```

- **0.8**: 平衡（推荐）
  ```
  The sun was shining brightly over the garden.
  ```

- **1.0**: 正常随机性
  ```
  The sun, which had been hidden, now appeared.
  ```

- **1.2+**: 更随机，更有创造性（可能不连贯）
  ```
  The sun?! Yes, a garden of stars!
  ```

### 批量生成模式

```bash
python generate.py batch
```

会生成多个不同温度的样本，方便对比。

---

## 可视化

### 查看训练曲线

```bash
python visualize_training.py
```

会生成：
1. **训练损失曲线图** (`training_curves.png`)
2. **过拟合检测图**
3. **训练摘要统计**

输出示例：
```
============================================================
  📊 训练摘要
============================================================

总迭代次数: 11
起始损失: 训练=4.1732, 验证=4.1698
最终损失: 训练=2.2145, 验证=2.2489

最佳验证损失: 2.2489 (迭代 5000)
损失改善: 46.1%
============================================================

✅ 训练曲线已保存到 training_curves.png
```

### 可视化要点

📈 **正常训练**:
- 训练损失和验证损失都稳定下降
- 两条曲线接近

⚠️ **过拟合**:
- 训练损失继续下降
- 验证损失开始上升

💡 **欠拟合**:
- 损失都很高
- 还有下降空间

---

## 常见问题

### Q1: 训练很慢怎么办？

**方法 1**: 减小批次大小
```python
# config.py
batch_size = 16  # 从 32 改成 16
```

**方法 2**: 减小上下文长度
```python
# config.py
block_size = 64  # 从 128 改成 64
```

**方法 3**: 减少训练步数
```python
# config.py
max_iters = 2000  # 从 5000 改成 2000
```

### Q2: 显存/内存不足

```python
# config.py
batch_size = 8      # 减小批次
block_size = 32     # 减小上下文
```

或使用 CPU:
```python
# config.py
device = "cpu"
```

### Q3: MPS 不可用（Mac M1/M2）

如果看到 MPS 错误，改用 CPU:
```python
# config.py
device = "cpu"
```

### Q4: 生成的文本完全乱码

**原因**: 训练不足

**解决**:
1. 继续训练更多步数
2. 检查损失是否在下降
3. 尝试降低温度参数（0.5）

### Q5: 损失不下降

**可能原因**:
1. 学习率太大或太小
2. 数据有问题
3. 代码有 bug

**调试步骤**:
```bash
# 1. 检查数据
python dataset.py

# 2. 检查模型
python model.py

# 3. 尝试调整学习率
# config.py: learning_rate = 1e-3  # 或 1e-4
```

### Q6: 如何恢复训练？

修改 `train.py`，添加加载检查点的代码：

```python
# 在创建模型后
if os.path.exists('checkpoints/best_model.pt'):
    checkpoint = torch.load('checkpoints/best_model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter']
    print(f"从迭代 {start_iter} 恢复训练")
```

---

## 🎓 理解 Bigram Model

### 工作原理

1. **Embedding Table**: `[vocab_size, vocab_size]`
   ```
   Token 'h' (id=3) → 向量 [0.2, 0.5, 0.1, ...]
   这个向量就是对所有下一个 token 的 logits
   ```

2. **Forward Pass**:
   ```
   输入: [B, T]
     ↓
   Embedding: [B, T, vocab_size]
     ↓
   每个位置都有对下一个 token 的预测
   ```

3. **Loss Calculation**:
   ```
   预测: [0.1, 0.7, 0.2]  (对 3 个 token 的概率)
   实际:  token 1
   Loss: -log(0.7) = 0.357
   ```

4. **Generation**:
   ```
   当前 token → 查表 → 得到 logits → softmax → 采样
   ```

### 局限性

- ❌ 只看当前一个 token
- ❌ 没有位置信息
- ❌ 无法理解长程依赖

例如:
```
"The cat sat on the ___"
Bigram 只看 "the"，无法知道应该填 "mat" 而不是 "dog"
```

---

## 🚀 下一步

完成 V1 后，可以继续：

- [ ] **V2**: 添加 Position Embedding
- [ ] **V3**: 添加 Self-Attention
- [ ] **V4**: 添加 Feed-Forward 网络
- [ ] **V5**: 多层 Transformer
- [ ] **V6**: 完整的 GPT

---

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [Andrej Karpathy - Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/)

---

## 💡 提示

1. **先跑通再优化**: 不要一开始就追求完美，先让代码跑起来
2. **小步快跑**: 每次只改一个地方，确保能跑
3. **多看日志**: 训练日志能告诉你很多信息
4. **多生成**: 经常生成文本看看效果
5. **享受过程**: 这是学习的最好方式！

---

🎉 祝你训练顺利！有问题随时查看这个文档。
