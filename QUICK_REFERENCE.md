# 🚀 myGPT V1 - 快速参考卡

## 一键启动菜单

```bash
./run.sh
```

---

## 📝 常用命令

### 快速测试
```bash
python quick_test.py          # 测试所有组件
python dataset.py             # 查看数据统计
python model.py               # 测试模型
```

### 训练和生成
```bash
python train.py               # 训练模型
python generate.py            # 交互式生成
python generate.py batch      # 批量生成
python test_generate.py       # 快速生成测试
```

### 可视化
```bash
python visualize_training.py  # 训练曲线
python process_data.py        # 数据分析
```

---

## 🎯 文件速查

### 核心代码
```
tokenizer.py    → 字符 tokenizer
dataset.py      → 数据加载
model.py        → Bigram 模型
train.py        → 训练脚本
generate.py     → 生成脚本
config.py       → 配置文件
```

### 重要文件
```
checkpoints/best_model.pt    → 最佳模型
training_log.json           → 训练日志
data/input.txt              → 训练数据
```

### 文档
```
README.md           → 项目主页 ⭐
GUIDE.md            → 使用指南
TRAINING_SUMMARY.md → 训练总结
FINAL_SUMMARY.md    → 最终总结
```

---

## ⚙️ 快速配置

编辑 `config.py`:

```python
# 基础配置
batch_size = 32          # 批次大小
block_size = 128         # 上下文长度
max_iters = 5000         # 训练步数
learning_rate = 3e-4     # 学习率
device = "mps"           # 设备 (mps/cuda/cpu)

# 高级配置 (V2+)
n_embd = 128            # 嵌入维度
n_head = 4              # 注意力头
n_layer = 4             # 层数
dropout = 0.2           # Dropout
```

---

## 🔧 快速修复

### 训练太慢
```python
batch_size = 16         # 减小批次
block_size = 64         # 减小上下文
```

### 内存不足
```python
device = "cpu"          # 使用 CPU
batch_size = 8          # 减小批次
```

### 生成乱码
```bash
# 方法1: 继续训练
max_iters = 10000

# 方法2: 降低温度
python generate.py
> temp 0.5
```

---

## 📊 关键指标

### 训练结果
```
初始损失: 4.62
最终损失: 3.09
改善: 33.2%
时间: 13秒
参数: 4,225
```

### 文件大小
```
模型: ~17 KB
数据: ~1.1 MB
日志: ~2 KB
```

---

## 🎮 交互式生成

```bash
python generate.py

# 命令:
提示文本      → 生成
temp 0.8     → 设置温度
quit         → 退出
```

### 温度效果
```
0.5  → 保守 (推荐开始)
0.8  → 平衡 (推荐)
1.0  → 正常
1.2+ → 随机
```

---

## 📈 评估模型

### 看训练日志
```bash
cat training_log.json | python -m json.tool
```

### 查看损失
```bash
python -c "
import json
with open('training_log.json') as f:
    logs = json.load(f)
    print(f'最终损失: {logs[-1][\"val_loss\"]:.4f}')
"
```

### 生成曲线
```bash
python visualize_training.py
open training_curves.png
```

---

## 🐛 调试技巧

### 检查数据
```python
from dataset import train_data, val_data, tokenizer
print(f"训练集: {len(train_data)}")
print(f"验证集: {len(val_data)}")
print(f"词汇表: {tokenizer.vocab_size}")
```

### 检查模型
```python
from model import BigramLanguageModel
model = BigramLanguageModel(65)
print(f"参数: {model.count_parameters()}")
```

### 检查设备
```python
import torch
print(f"MPS: {torch.backends.mps.is_available()}")
print(f"CUDA: {torch.cuda.is_available()}")
```

---

## 📚 学习路径

### 第一天
1. 阅读 `README.md`
2. 运行 `quick_test.py`
3. 查看 `dataset.py` 输出
4. 理解数据流程

### 第二天
1. 阅读 `model.py`
2. 理解 forward pass
3. 理解 generate 过程
4. 尝试修改模型

### 第三天
1. 运行 `train.py`
2. 观察训练过程
3. 尝试 `generate.py`
4. 调整温度参数

### 第四天
1. 阅读 `GUIDE.md`
2. 理解 Bigram 原理
3. 思考局限性
4. 准备 V2

---

## 🎯 下一步

### 短期
- [ ] 训练 10k 步
- [ ] 尝试不同温度
- [ ] 记录生成样本

### 中期 (V2)
- [ ] Position Embedding
- [ ] 更大的 embedding
- [ ] 更长的上下文

### 长期 (V3+)
- [ ] Self-Attention
- [ ] Multi-Head Attention
- [ ] Feed-Forward 层
- [ ] 完整 Transformer

---

## 💡 最佳实践

### 训练
✅ 从小模型开始
✅ 定期检查生成
✅ 保存最佳模型
✅ 记录训练日志

### 生成
✅ 先用低温度
✅ 逐步提高温度
✅ 比较不同效果
✅ 保存好的样本

### 调试
✅ 先测试组件
✅ 检查数据流
✅ 打印中间结果
✅ 逐步排查问题

---

## 🔗 快速链接

| 资源 | 链接 |
|------|------|
| 主文档 | [README.md](README.md) |
| 使用指南 | [GUIDE.md](GUIDE.md) |
| 训练总结 | [TRAINING_SUMMARY.md](TRAINING_SUMMARY.md) |
| 论文 | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) |
| 教程 | [Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) |

---

## 🆘 紧急救援

### 程序崩溃
```bash
# 检查错误日志
tail training_output.log

# 重新测试
python quick_test.py
```

### 模型丢失
```bash
# 检查备份
ls checkpoints/

# 重新训练
python train.py
```

### 环境问题
```bash
# 重新激活
source .venv/bin/activate

# 检查 PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## 📞 联系方式

遇到问题？查看:
1. `GUIDE.md` 的 FAQ 部分
2. `README.md` 的常见问题
3. 代码注释

---

<div align="center">

**快速参考 | V1.0 | 2026-05-28**

[🏠 主页](README.md) | [📖 指南](GUIDE.md) | [🎯 总结](FINAL_SUMMARY.md)

</div>
