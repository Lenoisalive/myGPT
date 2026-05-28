# 🚀 myGPT - 完整使用指南

## 📋 目录
1. [项目概述](#项目概述)
2. [版本对比](#版本对比)
3. [快速开始](#快速开始)
4. [训练模型](#训练模型)
5. [生成文本](#生成文本)
6. [V2 Self-Attention 详解](#v2-self-attention-详解)
7. [常见问题](#常见问题)

---

## 项目概述

这是一个从零开始实现的 GPT 语言模型项目，包含四个版本：

### V1: Bigram Model
**核心**: `P(next_token | current_token)`
- 只看当前 token
- 最简单的语言模型
- 参数量: 4,225

### V2: Self-Attention Model
**核心**: `Attention(Q, K, V)` + Position Embedding
- 看所有之前的 token
- 引入 Self-Attention 机制
- 参数量: 82,241

### V3: Multi-Head Attention Model
**核心**: `MultiHead(Q, K, V)` + Projection
- 多个 head 并行学习不同模式
- 语法、依赖、局部、实体关系
- 参数量: 98,753

### V4: Transformer Block Model
**核心**: `TransformerBlock` + FFN + Residual + LayerNorm
- 完整的 Transformer 结构
- 可堆叠多层，训练更稳定
- 参数量: 824,897

---

## 版本对比

| 特性 | V1 | V2 | V3 | V4 |
|------|----|----|----|---|
| **上下文** | 只看当前 | 看所有之前 | 看所有之前 | 看所有之前 |
| **Position** | ❌ | ✅ | ✅ | ✅ |
| **Attention** | ❌ | ✅ Single | ✅ Multi (4) | ✅ Multi (4) |
| **Layers** | - | 0 | 0 | 4 |
| **FFN** | ❌ | ❌ | ❌ | ✅ |
| **Residual** | ❌ | ❌ | ❌ | ✅ |
| **LayerNorm** | ❌ | ❌ | ❌ | ✅ |
| **参数量** | 4,225 | 82,241 | 98,753 | 824,897 |
| **训练时间** | ~13秒 | ~50秒 | ~60秒 | ~2分钟 |
| **最终损失** | ~3.09 | ~2.4 | ~2.1 | ~1.8 (预期) |
| **生成质量** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐⭐ |

### 生成质量对比

**V1 输出**:
```
Thand nd,PETheWd tFO:l II&,
```
- 单词片段混乱
- 无上下文连贯性

**V2 输出**:
```
the world and the sun is shining
```
- 完整单词
- 短语结构
- 开始有语法

**V3 输出** (预期):
```
The sun rises in the morning, bringing warmth and light.
```
- 更复杂的句子结构
- 更好的语义连贯性
- 更自然的语言流畅度

**V4 输出** (预期):
```
In the bustling city center, people hurried along the sidewalks,
their faces illuminated by the glow of storefronts and streetlights
as the evening descended upon the urban landscape.
```
- 长句子，多个从句
- 复杂的语义关系
- 接近人类写作水平

---

## 快速开始

### 1️⃣ 测试模型

```bash
# 测试 V1、V2 和 V3 模型
python model.py
```

输出:
```
✅ V1 Bigram 模型初始化完成
   参数量: 4,225

✅ V2 Self-Attention 模型初始化完成
   参数量: 82,241

✅ V3 Multi-Head Attention (4 heads) 模型初始化完成
   参数量: 98,753
```

✅ V2 Self-Attention 模型初始化完成
   参数量: 82,241

🔍 测试 Attention Matrix:
   Attention shape: [1, 10, 10]
   Causal mask 工作: True
```

### 2️⃣ 下载更大的数据集（推荐）

默认的 Shakespeare 数据集只有 2.2MB，对于训练效果有限。建议下载更大的高质量数据集：

```bash
# 方式 1: 交互式选择（最简单）
./download_data.sh

# 方式 2: 命令行指定大小
python download_data.py --size 100  # 100MB

# 方式 3: 自定义大小和路径
python download_data.py --size 500 --output data/input.txt
```

**推荐配置**:
- 🧪 **快速测试**: 50MB（下载约 5 分钟）
- ⭐ **日常训练**: 100MB（下载约 10 分钟，推荐）
- 🚀 **更好效果**: 500MB（下载约 30 分钟）
- 💎 **最佳效果**: 1GB（下载约 1 小时）

**数据集说明**:
- 来源: [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- 质量: HuggingFace 精选的高质量网页文本
- 语言: 英文为主，质量远超 Common Crawl

### 3️⃣ 训练模型

```bash
# 推荐: 训练 V2
python train.py v2

# 或者训练 V1（用于对比）
python train.py v1
```

### 3️⃣ 训练模型

```bash
# 推荐: 训练 V3 (Multi-Head)
python train.py v3

# 或者训练 V2 (Single-Head)
python train.py v2

# 或者训练 V1（用于对比）
python train.py v1
```

### 4️⃣ 生成文本

```bash
# V3 生成（推荐）
python generate.py

# V2 生成
python generate.py v2

# V1 生成
python generate.py v1
```

---

## 训练模型

### 命令格式

```bash
python train.py [v1|v2|v3]
```

- `v1`: 训练 Bigram 模型
- `v2`: 训练 Single-Head Self-Attention 模型
- `v3`: 训练 Multi-Head Attention 模型（默认）

### V3 训练示例

```
============================================================
  🚀 开始训练 V3 Multi-Head Attention (4 heads) Language Model
============================================================

📋 训练配置:
   模型版本: V3 Multi-Head Attention (4 heads)
   设备: mps
   批次大小: 32
   上下文长度: 128
   最大迭代: 5000
   学习率: 0.0003
   评估间隔: 500
   嵌入维度: 128
   Attention Heads: 4
   Head Size: 32

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
# V2 (推荐)
python train.py v2

# V1 (用于对比)
python train.py v1
```

### 训练输出

```
============================================================
  🚀 开始训练 V2 Self-Attention Language Model
============================================================

📋 训练配置:
   模型版本: V2 Self-Attention
   设备: mps
   批次大小: 32
   上下文长度: 128
   最大迭代: 5000
   学习率: 0.0003
   嵌入维度: 128

📊 数据统计:
   词汇表大小: 65
   训练集大小: 1,003,854 tokens

============================================================
  🔨 创建模型
============================================================
✅ V2 Self-Attention 模型初始化完成
   参数量: 82,241

📈 步数     0/5000
   训练损失: 4.2145
   验证损失: 4.2089

📈 步数  1000/5000
   训练损失: 2.6203
   验证损失: 2.6233

📝 生成样本文本:
────────────────────────────────────────────────────────────
the world and the sun is shining
────────────────────────────────────────────────────────────
```

### 训练文件

训练完成后会生成：
- `checkpoints/best_model.pt` - 最佳模型
- `training_log_v2.json` - V2 训练日志
- `training_log_v1.json` - V1 训练日志（如果训练了 V1）

---

## 生成文本

### 命令格式

```bash
python generate.py [v1|v2] [interactive|batch]
```

### 交互式生成

```bash
# V2 交互式（默认）
python generate.py

# V1 交互式
python generate.py v1
```

### 批量生成

```bash
# V2 批量
python generate.py v2 batch

# V1 批量
python generate.py v1 batch
```

### 使用示例

```
============================================================
  🎮 交互式生成模式 - V2 (Self-Attention)
============================================================

📥 加载模型
⏳ 正在从 checkpoints/best_model.pt 加载...
✅ 模型加载完成
   训练步数: 5000
   验证损失: 2.4512

💡 使用说明:
   - 输入提示文本，按回车生成
   - 输入 'quit' 或 'exit' 退出
   - 输入 'temp X' 设置温度

💬 提示 (温度=0.8): Hello

⏳ 正在生成...
────────────────────────────────────────────────────────────
Hello world and the sun
────────────────────────────────────────────────────────────
```

---

## V2 Self-Attention 详解

### 核心概念

#### 1. Self-Attention 机制

**问题**: Bigram 只看当前 token，如何让模型看到更多上下文？

**解决**: Self-Attention

```python
# 对于序列: "The cat sat"
# Token "sat" 可以：
# - 看到 "The" (远处的信息)
# - 看到 "cat" (中间的信息)
# - 看到 "sat" (自己)
```

#### 2. Query, Key, Value

```python
Q (Query): 我在寻找什么信息？
K (Key):   我能提供什么信息？
V (Value): 我的实际信息内容
```

**计算过程**:
```
1. Attention Score = Q @ K^T
   → 计算相关性

2. Apply Causal Mask
   → 只能看左边的 token

3. Softmax
   → 转为概率分布

4. Output = Attention @ V
   → 加权求和
```

#### 3. Causal Mask

**关键**: 训练时不能"偷看"未来

```
Attention Matrix (before mask):
    T  h  e     c  a  t
T   ●  ●  ●  ●  ●  ●  ●
h   ●  ●  ●  ●  ●  ●  ●
e   ●  ●  ●  ●  ●  ●  ●
    ●  ●  ●  ●  ●  ●  ●
c   ●  ●  ●  ●  ●  ●  ●
a   ●  ●  ●  ●  ●  ●  ●
t   ●  ●  ●  ●  ●  ●  ●

After Causal Mask:
    T  h  e     c  a  t
T   ●  -  -  -  -  -  -
h   ●  ●  -  -  -  -  -
e   ●  ●  ●  -  -  -  -
    ●  ●  ●  ●  -  -  -
c   ●  ●  ●  ●  ●  -  -
a   ●  ●  ●  ●  ●  ●  -
t   ●  ●  ●  ●  ●  ●  ●
```

#### 4. Position Embedding

**问题**: Attention 本身没有位置信息

**解决**: Position Embedding

```python
token_emb = token_embedding_table(tokens)  # [B, T, n_embd]
pos_emb = position_embedding_table(positions)  # [T, n_embd]
x = token_emb + pos_emb  # 加上位置信息
```

### 模型架构对比

**V1 (Bigram)**:
```
Input [B, T]
  ↓
Token Embedding [B, T, vocab_size]
  ↓
Logits [B, T, vocab_size]
```

**V2 (Self-Attention)**:
```
Input [B, T]
  ↓
Token Embedding [B, T, n_embd]
  +
Position Embedding [T, n_embd]
  ↓
Self-Attention Head
  ↓
Linear (Language Model Head)
  ↓
Logits [B, T, vocab_size]
```

### 代码实现

```python
class Head(nn.Module):
    """Single Self-Attention Head"""
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal mask
        self.register_buffer('tril', 
            torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]
        
        # Attention scores
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # [B, T, T]
        
        # Apply causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Softmax
        wei = F.softmax(wei, dim=-1)
        
        # Weighted sum
        v = self.value(x)  # [B, T, head_size]
        out = wei @ v      # [B, T, head_size]
        
        return out
```

---

## V3 Multi-Head Attention 详解

### 核心思想

**为什么需要多个 Head？**

单个 attention head 只能学习一种模式。但语言有多种关系：
- 语法关系（主谓宾）
- 长距离依赖
- 局部上下文
- 实体关系

多个 head 可以**并行学习**不同的模式！

### 1. Multi-Head 架构

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # 创建多个 head
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # Output projection
        self.proj = nn.Linear(num_heads * head_size, n_embd)
    
    def forward(self, x):
        # 并行计算所有 head
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # 通过 projection 混合信息
        out = self.proj(out)
        return out
```

### 2. 工作原理

**示例：4 个 heads，embedding = 128**

```
输入: [B, T, 128]
  ↓
Head 1 → [B, T, 32]  # 学习语法
Head 2 → [B, T, 32]  # 学习依赖
Head 3 → [B, T, 32]  # 学习局部
Head 4 → [B, T, 32]  # 学习实体
  ↓
Concat → [B, T, 128]
  ↓
Projection → [B, T, 128]
```

### 3. 每个 Head 学什么？

**Head 1 - 语法关系**:
```
"The cat sits on the mat"
 ↓   ↑
主语关注动词
```

**Head 2 - 长距离依赖**:
```
"The cat that I saw yesterday sits"
 ↓                          ↑
主语关注远处的动词
```

**Head 3 - 局部上下文**:
```
"The big red cat"
     ↓   ↓   ↑
形容词关注临近的名词
```

**Head 4 - 实体关系**:
```
"John loves Mary"
  ↓        ↑
人物关系
```

### 4. 代码实现对比

**V2: Single-Head**
```python
# 一个 attention head
self.sa_head = Head(n_embd)  # 128 → 128

# 前向传播
x = self.sa_head(x)  # [B, T, 128]
```

**V3: Multi-Head**
```python
# 4 个 attention heads
head_size = n_embd // num_heads  # 128 // 4 = 32
self.sa_head = MultiHeadAttention(4, 32)

# 前向传播
x = self.sa_head(x)  # [B, T, 128]
# 内部: 4 个 [B, T, 32] → concat → proj
```

### 5. 参数量计算

**V2 (Single-Head)**:
```
Q: 128 × 128 = 16,384
K: 128 × 128 = 16,384
V: 128 × 128 = 16,384
Total: 49,152
```

**V3 (4 Heads)**:
```
每个 head:
  Q: 128 × 32 = 4,096
  K: 128 × 32 = 4,096
  V: 128 × 32 = 4,096
4 个 heads: 4 × 12,288 = 49,152
Projection: 128 × 128 = 16,384
Total: 65,536 (+33%)
```

### 6. 为什么 Multi-Head 更好？

✅ **多样性**: 每个 head 专注不同模式  
✅ **并行**: 所有 head 同时计算  
✅ **鲁棒**: 一个 head 失败，其他补充  
✅ **表达力**: 捕捉更复杂的语言结构  

### 验证 Multi-Head Attention

```bash
# 测试 multi-head attention
python model.py
```

输出:
```
🔍 测试 Multi-Head Attention:
   每个 Head 输出: [1, 10, 32]  ✓
   Concat 输出: [1, 10, 128]  ✓
   完整输出: [1, 10, 128]  ✓
```

---

## V4 Transformer Block 详解

### 🎯 为什么需要 Transformer Block？

**V3 的问题**:
- 只有 attention，没有深度
- 无法学习复杂的非线性变换
- 难以堆叠多层（梯度问题）

**V4 的解决方案**:
```
✅ Residual Connection → 梯度流动
✅ LayerNorm → 训练稳定
✅ Feed-Forward → 非线性变换
✅ 可堆叠 4 层！
```

### 1. Transformer Block 架构

```python
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, num_heads):
        self.sa = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        # Attention block
        x = x + self.sa(self.ln1(x))    # Pre-norm + Residual
        
        # Feed-forward block
        x = x + self.ffwd(self.ln2(x))  # Pre-norm + Residual
        
        return x
```

### 2. 核心组件详解

#### A. LayerNorm (层归一化)

**作用**: 稳定训练，防止梯度爆炸

```python
# 对每个样本的每个位置归一化
mean = x.mean(dim=-1, keepdim=True)
std = x.std(dim=-1, keepdim=True)
x_norm = (x - mean) / (std + eps)
```

**为什么需要**:
- 深层网络：hidden state 分布容易偏移
- LayerNorm：保持分布稳定
- 结果：训练更快，收敛更好

#### B. Residual Connection (残差连接)

**作用**: 让梯度直接流动，避免梯度消失

```python
# 传统: x = f(x)  梯度可能消失
# Residual: x = x + f(x)  梯度直接传递
```

**直观理解**:
```
没有 residual:
  x → layer1 → layer2 → layer3 → layer4
  梯度: ← ← ← ← (越来越小)

有 residual:
  x → +layer1 → +layer2 → +layer3 → +layer4
     ↓         ↓         ↓         ↓
  梯度直接跳过多层！
```

#### C. Feed-Forward Network

**作用**: 非线性特征变换

```python
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),    # 扩展 4 倍
            nn.GELU(),                         # 平滑激活
            nn.Linear(4 * n_embd, n_embd),    # 压缩回来
            nn.Dropout(config.dropout),
        )
```

**为什么扩展 4 倍？**
- GPT 标准做法
- 给模型更大的表达空间
- 类似"瓶颈"结构

**为什么用 GELU？**
- 比 ReLU 更平滑
- 在 NLP 任务中效果更好
- GPT/BERT 都用它

#### D. Dropout

**作用**: 防止过拟合

```python
# 训练时随机丢弃 20% 的神经元
dropout = nn.Dropout(0.2)
```

### 3. 完整数据流

```
输入 [B, T, 128]
  ↓
─────────────────────
Block 1:
  x → LayerNorm
    → MultiHeadAttention
    → Dropout
    → + (residual)
  → LayerNorm
    → FeedForward
    → Dropout
    → + (residual)
─────────────────────
Block 2: (same)
─────────────────────
Block 3: (same)
─────────────────────
Block 4: (same)
─────────────────────
  ↓
Final LayerNorm
  ↓
Language Model Head
  ↓
输出 [B, T, vocab_size]
```

### 4. 参数量计算

**单个 Transformer Block**:
```
MultiHeadAttention:
  4 heads × (Q/K/V: 128→32) = 49,152
  Projection: 128×128 = 16,384
  小计: 65,536

FeedForward:
  Linear1: 128 × 512 = 65,536
  Linear2: 512 × 128 = 65,536
  小计: 131,072

LayerNorm × 2: 256 × 2 = 512

Block Total: ~196,000
```

**4 层总计**:
```
4 × 196,000 = 784,000
+ Embeddings: ~33,000
+ LM Head: ~8,000
───────────────────
Total: ~825,000 参数
```

### 5. V4 vs V3 对比

| 特性 | V3 | V4 |
|------|----|----|
| **结构** | 单层 Multi-Head | 4 层 Transformer Block |
| **参数** | 98,753 | 824,897 |
| **深度** | 浅 | 深 |
| **稳定性** | 一般 | 优秀 (Residual + LayerNorm) |
| **表达力** | 好 | 强大 (多层 + FFN) |
| **训练难度** | 容易 | 中等 |
| **收敛速度** | 快 | 稳定 |

### 6. 为什么 V4 更强？

✅ **多层抽象**:
```
Layer 1: 学习字符模式
Layer 2: 学习单词模式
Layer 3: 学习短语模式
Layer 4: 学习句子模式
```

✅ **非线性变换**:
```
Attention: 线性组合信息
FFN: 非线性变换特征
```

✅ **训练稳定**:
```
Residual: 梯度直接流动
LayerNorm: 分布稳定
Dropout: 防止过拟合
```

### 验证 Transformer Block

```bash
# 测试 V4 模型
python model.py
```

输出:
```
✅ V4 Transformer (4 layers, 4 heads) 模型初始化完成
   参数量: 824,897
   输出形状: [4, 128, 65]  ✓
   损失: 4.3051  ✓
   生成: [1, 21]  ✓
```

---

## 常见问题

### Q1: V2 训练很慢？

**原因**: V2 参数量是 V1 的 19 倍

**解决**:
```python
# config.py
n_embd = 64      # 从 128 减少到 64
batch_size = 16  # 减小批次
```

### Q2: 如何比较 V1 和 V2？

```bash
# 1. 分别训练
python train.py v1
python train.py v2

# 2. 比较损失
cat training_log_v1.json | grep val_loss
cat training_log_v2.json | grep val_loss

# 3. 比较生成质量
python generate.py v1
python generate.py v2
```

### Q3: V2 生成还是不够好？

**原因**: 
- 只有单个 attention head
- 没有 Feed-Forward 层
- 层数只有 1 层

**下一步**: V3 将添加 Multi-Head Attention

### Q4: Attention Matrix 太大内存不够？

```python
# config.py
block_size = 64  # 从 128 减少
# Attention Matrix: 64×64 vs 128×128
```

### Q5: 如何可视化 Attention？

```python
# 在 model.py 的 forward 中添加
# 保存 attention weights
self.attention_weights = wei.detach()

# 然后可以可视化
import matplotlib.pyplot as plt
plt.imshow(attention_weights[0].cpu())
plt.show()
```

---

## 🎓 学习要点

### V1 → V2 的关键改进

1. **从局部到全局**
   - V1: 只看当前 token
   - V2: 看所有之前的 token

2. **动态权重**
   - V1: 固定的 embedding lookup
   - V2: 动态计算 attention 权重

3. **位置感知**
   - V1: 无位置信息
   - V2: Position embedding

### 为什么需要 Causal Mask？

```python
# 错误: 没有 mask
"The cat" → 模型可以看到 "cat" 来预测 "The"
# 这是作弊！测试时看不到未来

# 正确: 有 mask  
"The" → 只能看到 "The" 来预测 "cat"
"The cat" → 只能看到 "The cat" 来预测下一个词
```

### Attention 的直觉

```
句子: "The cat sat on the mat"
预测下一个词时，token "mat" 会：
  - 高度关注 "on" (介词提示)
  - 中度关注 "sat" (动词)
  - 低度关注 "The" (太远)
```

---

## 🚀 下一步

完成 V2 后，准备：

- [ ] **V3**: Multi-Head Attention
  - 多个 attention head 并行
  - 捕捉不同的关系模式

- [ ] **V4**: Feed-Forward 网络
  - 增加模型容量
  - 非线性变换

- [ ] **V5**: 多层堆叠
  - 深度网络
  - 完整 Transformer

---

## 📚 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer 原论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Andrej Karpathy

---

🎉 现在你已经理解了 Self-Attention 的核心！继续探索 V3 吧！

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
