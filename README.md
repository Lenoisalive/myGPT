# 🤖 myGPT - 从零实现 GPT 语言模型

一个从零开始、逐步构建的 GPT 语言模型项目。

## 🎯 项目目标

通过逐步实现，深入理解 GPT 的工作原理：
1. ✅ **V1**: Bigram Language Model（当前版本）
2. 🔄 **V2**: 添加 Position Embedding
3. 🔄 **V3**: 添加 Self-Attention
4. 🔄 **V4**: 添加 Feed-Forward 网络
5. 🔄 **V5**: 完整的 Transformer 架构

---

## ✅ V1: Bigram Language Model - 已完成！

### 核心思想
```
P(next_token | current_token)
```

最简单的语言模型：给定当前 token，预测下一个 token。

### 训练结果

| 指标 | 数值 |
|------|------|
| 初始损失 | 4.62 |
| 最终损失 | 3.09 |
| 改善程度 | 33.2% ↓ |
| 训练时长 | 13秒 |
| 模型参数 | 4,225 |

### 生成示例

**温度 = 0.5**:
```
Thand nd,PETheWd tFO:l II&,
WLLEUzhav
```

**温度 = 0.8**:
```
We.Nve fyrs ald thwGangugo!purso!
```

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux
# 或
.venv\Scripts\activate     # Windows

# 检查 PyTorch
python -c "import torch; print(torch.__version__)"
```

### 2. 快速测试

```bash
python quick_test.py
```

### 3. 训练模型（已完成）

```bash
python train.py
```

### 4. 生成文本

```bash
# 交互式模式
python generate.py

# 批量生成
python generate.py batch
```

### 5. 可视化训练

```bash
python visualize_training.py
```

---

## 📁 项目结构

```
myGPT/
├── 核心代码
│   ├── tokenizer.py          # 字符级 tokenizer
│   ├── dataset.py            # 数据处理
│   ├── model.py              # Bigram 模型
│   ├── train.py              # 训练脚本
│   ├── generate.py           # 生成脚本
│   └── config.py             # 配置文件
│
├── 工具脚本
│   ├── quick_test.py         # 快速测试
│   ├── test_generate.py      # 生成测试
│   ├── process_data.py       # 数据分析
│   └── visualize_training.py # 训练可视化
│
├── 文档
│   ├── README.md             # 本文件
│   ├── README_V1.md          # V1 详细说明
│   ├── GUIDE.md              # 完整使用指南
│   ├── TRAINING_SUMMARY.md   # 训练总结
│   └── PROJECT_COMPLETE.md   # 项目完成报告
│
├── 数据和模型
│   ├── data/
│   │   └── input.txt         # 训练数据
│   ├── checkpoints/
│   │   ├── best_model.pt     # 最佳模型
│   │   └── final_model.pt    # 最终模型
│   └── training_log.json     # 训练日志
│
└── 配置
    └── config.py             # 超参数配置
```

---

## 📚 文档导航

| 文档 | 内容 |
|------|------|
| [README_V1.md](README_V1.md) | V1 版本详细说明 |
| [GUIDE.md](GUIDE.md) | 完整使用指南 |
| [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) | 训练过程和结果 |
| [PROJECT_COMPLETE.md](PROJECT_COMPLETE.md) | 项目完成报告 |

---

## 🎓 核心概念

### Bigram Model

```python
# 核心：一个简单的 embedding table
token_embedding = nn.Embedding(vocab_size, vocab_size)

# 每个 token 直接映射到下一个 token 的 logits
logits = token_embedding(current_token)  # [vocab_size]
next_token = sample(softmax(logits))
```

### 训练流程

```
1. 加载数据 → Tokenize → [token_ids]
2. 创建批次 → x=[1,2,3,4], y=[2,3,4,5]
3. 前向传播 → logits = model(x)
4. 计算损失 → loss = cross_entropy(logits, y)
5. 反向传播 → loss.backward()
6. 更新参数 → optimizer.step()
```

### 生成流程

```
start_token → lookup_table → logits → softmax → sample
     ↓             ↓            ↓         ↓         ↓
   [34]    →    [...]    →   [...]   →  [...]   → [23]
                                                     ↓
                                              append & repeat
```

---

## ⚙️ 配置说明

编辑 `config.py` 修改超参数：

```python
# 训练配置
batch_size = 32          # 批次大小
block_size = 128         # 上下文长度
max_iters = 5000         # 训练步数
learning_rate = 3e-4     # 学习率
eval_interval = 500      # 评估间隔

# 设备配置
device = "mps"           # mps/cuda/cpu

# 模型配置（V2+ 使用）
n_embd = 128            # 嵌入维度
n_head = 4              # 注意力头数
n_layer = 4             # 层数
dropout = 0.2           # Dropout 率
```

---

## 🎮 使用示例

### 交互式生成

```bash
$ python generate.py

============================================================
  🎮 交互式生成模式
============================================================
💡 使用说明:
   - 输入提示文本，按回车生成
   - 输入 'quit' 或 'exit' 退出
   - 输入 'temp X' 设置温度

💬 提示 (温度=0.8): Hello world

⏳ 正在生成...
────────────────────────────────────────────────────────────
Hello world and the sun...
────────────────────────────────────────────────────────────

💬 提示 (温度=0.8): temp 0.5
✅ 温度已设置为 0.5

💬 提示 (温度=0.5): The 
────────────────────────────────────────────────────────────
The sun is shining...
────────────────────────────────────────────────────────────

💬 提示 (温度=0.5): quit
👋 再见!
```

---

## 📈 训练可视化

运行 `python visualize_training.py` 生成训练曲线：

```
============================================================
  📊 训练摘要
============================================================

总迭代次数: 11
起始损失: 训练=4.6331, 验证=4.6184
最终损失: 训练=3.0810, 验证=3.0866

最佳验证损失: 3.0866 (迭代 4999)
损失改善: 33.2%

✅ 训练曲线已保存到 training_curves.png
```

---

## 🔧 常见问题

### Q: 训练很慢？
A: 减小 `batch_size` 或 `block_size`

### Q: 内存不足？
A: 设置 `device = "cpu"` 或减小批次大小

### Q: 生成的文本乱码？
A: 
1. 继续训练更多步数
2. 降低温度参数（0.5）
3. 检查损失是否在下降

### Q: MPS 不可用？
A: 设置 `device = "cpu"` 或 `"cuda"`（如果有 NVIDIA GPU）

---

## 🎯 V1 学习要点

### ✅ 已学会的

1. **Tokenization**: 文本 ↔ 数字转换
2. **Dataset**: 批次生成和数据处理
3. **Training Loop**: 完整的训练流程
4. **Loss Function**: 交叉熵损失
5. **Text Generation**: 自回归生成
6. **Model Evaluation**: 训练/验证集评估

### ❌ V1 的局限

1. 只能看当前一个 token
2. 没有位置信息
3. 无法理解上下文
4. 没有 attention 机制

**→ 这些将在 V2+ 中解决！**

---

## 🚀 下一步：V2

### 计划添加

1. **Position Embedding**
   - 让模型理解 token 的位置
   
2. **更大的 Context Window**
   - 增加 `block_size`
   
3. **更好的 Embedding**
   - 使用 `n_embd` 维度
   - 分离输入和输出 embedding

---

## 📊 性能基准

| 模型版本 | 参数量 | 训练时间 | 最终损失 | 生成质量 |
|---------|--------|---------|---------|---------|
| V1 (Bigram) | 4,225 | 13秒 | 3.09 | ⭐⭐ |
| V2 (Position) | TBD | TBD | TBD | TBD |
| V3 (Attention) | TBD | TBD | TBD | TBD |

---

## 🤝 贡献

这是一个学习项目，欢迎：
- 提出改进建议
- 报告 bug
- 分享使用经验

---

## 📖 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy 视频教程
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程

---

## 📝 License

MIT License

---

## 🎉 项目状态

**V1: Bigram Language Model** - ✅ 完成

- ✅ 模型实现
- ✅ 训练完成
- ✅ 生成功能
- ✅ 文档完善
- ✅ 测试通过

**准备开始 V2！** 🚀

---

**最后更新**: 2026年5月28日  
**当前版本**: V1 - Bigram Language Model  
**状态**: ✅ Production Ready

---

<div align="center">

**Made with ❤️ for Learning**

[📖 使用指南](GUIDE.md) | [📊 训练总结](TRAINING_SUMMARY.md) | [🎯 完成报告](PROJECT_COMPLETE.md)

</div>
