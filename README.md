# 🤖 myGPT - 从零实现 GPT 语言模型

一个从零开始、逐步构建的 GPT 语言模型项目。

## 🎯 项目目标

通过逐步实现，深入理解 GPT 的工作原理：
1. ✅ **V1**: Bigram Language Model（已完成）
2. ✅ **V2**: Single-Head Self-Attention（已完成）
3. ✅ **V3**: Multi-Head Attention（已完成）
4. ✅ **V4**: Transformer Block（已完成）
5. 🔄 **V5**: 完整的 GPT 架构

---

## ✅ V4: Transformer Block - 已完成！

### 🎉 真正的 Transformer 来了！

**从 V3 到 V4 的质变**:
```
V3: 只有 Multi-Head Attention
V4: 完整 Transformer Block
    - LayerNorm (稳定训练)
    - Residual Connection (梯度流动)
    - Feed-Forward Network (特征变换)
    - Dropout (防止过拟合)
    - 可堆叠多层！
```

### 核心突破

**Transformer Block 结构**:
```python
x → LayerNorm → MultiHeadAttention → Residual
  → LayerNorm → FeedForward → Residual
```

**为什么需要这些组件？**

| 组件 | 作用 | 解决的问题 |
|------|------|------------|
| **LayerNorm** | 归一化 hidden state | 梯度爆炸/消失 |
| **Residual** | 跳跃连接 `x = x + f(x)` | 深层网络退化 |
| **FFN** | 两层 MLP (扩展 4 倍) | 非线性特征变换 |
| **Dropout** | 随机丢弃神经元 | 过拟合 |

### 技术实现

1. **Feed-Forward Network**
   ```python
   Linear(n_embd → 4*n_embd)  # 扩展
   → GELU                      # 平滑激活
   → Linear(4*n_embd → n_embd)  # 压缩
   → Dropout
   ```

2. **Transformer Block**
   ```python
   # Attention block
   x = x + dropout(attention(layernorm(x)))
   
   # FFN block  
   x = x + dropout(ffn(layernorm(x)))
   ```

3. **可堆叠多层**
   - 4 层 Transformer blocks
   - 每层独立学习不同抽象级别
   - 最后加 LayerNorm

### 参数量对比

| 版本 | 参数量 | Heads | Layers | 训练难度 |
|------|--------|-------|--------|----------|
| V1 | 4,225 | - | - | ⭐ |
| V2 | 82,241 | 1 | 0 | ⭐⭐ |
| V3 | 98,753 | 4 | 0 | ⭐⭐⭐ |
| **V4** | **824,897** | **4** | **4** | ⭐⭐⭐⭐ |

V4 参数量暴增到 **82 万**！但训练更稳定。

---

## ✅ V3: Multi-Head Attention

### 核心突破

**从 V2 到 V3 的关键变化**:
```
V2 (Single-Head):  1个 attention 学习 1 种模式
V3 (Multi-Head):   4个 attention 并行学习多种模式
```

### 多头注意力的威力

不同的 head 可以专注于学习不同的语言模式：
- **Head 1**: 语法关系（主谓宾）
- **Head 2**: 长距离依赖
- **Head 3**: 局部上下文
- **Head 4**: 实体关系

然后通过 **projection layer** 融合所有信息！

### 技术实现

1. **并行计算多个 Attention Heads**
   ```python
   heads = [Head(head_size) for _ in range(num_heads)]
   out = torch.cat([h(x) for h in heads], dim=-1)
   ```

2. **输出投影**
   ```python
   self.proj = nn.Linear(num_heads * head_size, n_embd)
   ```

3. **参数量对比**
   - V1: 4,225
   - V2: 82,241
   - **V3**: 98,753 (+20% vs V2)

---

## ✅ V2: Single-Head Self-Attention

### 核心改进

**从 V1 到 V2 的关键变化**:
```
V1 (Bigram):  只看当前 token → 预测下一个
V2 (Attention): 看所有之前的 token → 预测下一个
```

### 新增功能

1. **Self-Attention 机制**
   - Query/Key/Value 投影
   - Scaled Dot-Product Attention
   - Causal Masking (只看左边)

2. **Position Embedding**
   - 让模型知道 token 的位置
   - 与 token embedding 相加

3. **更大的 Embedding 空间**
   - 从 `[vocab_size, vocab_size]` → `[vocab_size, n_embd]`
   - 参数量: 4,225 → 82,241 (增加 ~19倍)

### 训练结果对比

| 指标 | V1 | V2 | V3 | V4 |
|------|----|----|----|----|
| **参数量** | 4,225 | 82,241 | 98,753 | 824,897 |
| **Attention Heads** | - | 1 | 4 | 4 |
| **Layers** | - | 0 | 0 | 4 |
| **初始损失** | 4.62 | ~4.2 | ~4.2 | ~4.3 |
| **预期最终损失** | 3.09 | ~2.4 | ~2.1 | ~1.8 |
| **训练时间** | 13秒 | ~50秒 | ~60秒 | ~2分钟 |
| **训练稳定性** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **生成质量** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐⭐ |

### 生成质量对比

**V1 (Bigram) - 迭代 5000**:
```
Thand nd,PETheWd tFO:l II&,
WLLEUzhav
```
❌ 单词混乱，无上下文

**V2 (Self-Attention) - 迭代 2000**:
```
Tmt,d y sheelot benoo ye what s t sa eand 
averates anecI pane, thart,
Tow arifor menluthoise papoo
```
✅ 开始出现短语结构，单词更完整

---

## ✅ V1: Bigram Language Model

### 核心思想
```
P(next_token | current_token)
```

最简单的语言模型：给定当前 token，预测下一个 token。

### 训练结果对比

| 指标 | V1 (Bigram) | V2 (Self-Attention) | 改善 |
|------|-------------|---------------------|------|
| 参数量 | 4,225 | 82,241 | +19倍 |
| 初始损失 | 4.62 | 4.17 | - |
| 最终损失 | 3.09 | 2.40 | ↓ 22% |
| 训练时间 | 13秒 | 44秒 | +3.4倍 |
| 生成质量 | ⭐⭐ | ⭐⭐⭐⭐ | 显著提升 |

### 生成质量对比

**V1 (Bigram) - 迭代 5000**:
```
Thand nd,PETheWd tFO:l II&,
WLLEUzhav
```
❌ 单词混乱，无上下文

**V2 (Self-Attention) - 迭代 5000**:
```
I Maatt ton loud thee me that in tents 
it the tothand telo she alivet the ard 
tond the te the outre ing thin
```
✅ 更完整的单词，开始有短语结构

**V2 温度 = 0.8**:
```
BARICBLADARI:
T INou thahe lat,
Yer lall wadond two riy,

ORAUNESTin macou warfa sthe anned,
Thing the mot wout be soe ou chahe
```
✅ 出现了类似名字、对话的结构

---

## ✅ V1: Bigram Language Model

### 核心思想
```
P(next_token | current_token)
```

最简单的语言模型：给定当前 token，预测下一个 token。

### V1 训练结果

| 指标 | 数值 |
|------|------|
| 初始损失 | 4.62 |
| 最终损失 | 3.09 |
| 改善程度 | 33.2% ↓ |
| 训练时长 | 13秒 |
| 模型参数 | 4,225 |

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 激活虚拟环境
source .venv/bin/activate  # macOS/Linux

# 检查 PyTorch
python -c "import torch; print(torch.__version__)"
```

### 2. 下载更大的数据集（推荐）

```bash
# 方式 1: 使用交互式脚本（推荐）
./download_data.sh

# 方式 2: 直接指定大小
python download_data.py --size 100  # 下载 100MB 数据

# 常用大小建议:
# 50MB   - 快速测试
# 100MB  - 日常训练（推荐）
# 500MB  - 更好效果
# 1024MB - 最佳效果
```

**数据集来源**: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) - HuggingFace 的高质量网页文本数据集

### 3. 快速测试

```bash
# 测试所有版本（V1、V2、V3、V4）
python model.py
```

### 4. 训练模型

```bash
# 训练 V4 (Transformer) - 推荐！
python train.py v4

# 或训练 V3 (Multi-Head)
python train.py v3

# 或训练 V2 (Single-Head)
python train.py v2

# 或训练 V1 (Bigram)
python train.py v1
```

### 5. 生成文本

```bash
# V4 交互式生成（默认，推荐）
python generate.py

# V3 交互式生成
python generate.py v3

# V2 交互式生成
python generate.py v2

# V1 交互式生成
python generate.py v1

# V4 批量生成
python generate.py v4 batch
```

### 6. 比较模型

```bash
# 训练不同版本并对比
python train.py v1
python train.py v2
python train.py v3
python train.py v4

# 对比生成质量
python generate.py v1
python generate.py v2
python generate.py v3
python generate.py v4
```

---

## 📁 项目结构

```
myGPT/
├── 核心代码
│   ├── tokenizer.py          # 字符级 tokenizer
│   ├── dataset.py            # 数据处理
│   ├── model.py              # V1 + V2 模型
│   ├── train.py              # 训练脚本（支持 v1/v2）
│   ├── generate.py           # 生成脚本（支持 v1/v2）
│   └── config.py             # 配置文件
│
├── 工具脚本
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
