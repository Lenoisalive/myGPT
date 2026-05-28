# quick_test.py
"""
快速测试所有组件
"""

import torch
from model import BigramLanguageModel
from tokenizer import CharTokenizer
import config

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

print_section("🧪 快速测试所有组件")

# 1. 测试 Tokenizer
print("\n1️⃣  测试 Tokenizer")
print("─"*60)
text = "hello world"
tokenizer = CharTokenizer(text)
print(f"✅ 词汇表大小: {tokenizer.vocab_size}")
print(f"✅ 字符集: {tokenizer.chars}")

encoded = tokenizer.encode("hello")
print(f"✅ 编码 'hello': {encoded}")

decoded = tokenizer.decode(encoded)
print(f"✅ 解码: '{decoded}'")

# 2. 测试 Model
print("\n2️⃣  测试 Bigram Model")
print("─"*60)
model = BigramLanguageModel(tokenizer.vocab_size)
print(f"✅ 模型参数量: {model.count_parameters():,}")

# 前向传播
idx = torch.randint(0, tokenizer.vocab_size, (2, 8))
targets = torch.randint(0, tokenizer.vocab_size, (2, 8))
logits, loss = model(idx, targets)
print(f"✅ 前向传播成功")
print(f"   输入: {idx.shape}")
print(f"   输出: {logits.shape}")
print(f"   损失: {loss.item():.4f}")

# 生成
context = torch.zeros((1, 1), dtype=torch.long)
generated = model.generate(context, max_new_tokens=20)
print(f"✅ 生成成功: {generated.shape}")
print(f"   生成的 tokens: {generated[0].tolist()}")

# 3. 测试 Dataset
print("\n3️⃣  测试 Dataset")
print("─"*60)
try:
    from dataset import get_batch, train_data, val_data
    print(f"✅ 训练集大小: {len(train_data):,}")
    print(f"✅ 验证集大小: {len(val_data):,}")
    
    x, y = get_batch('train')
    print(f"✅ 获取批次成功")
    print(f"   x: {x.shape}")
    print(f"   y: {y.shape}")
    print(f"   设备: {x.device}")
except Exception as e:
    print(f"⚠️  Dataset 测试失败: {e}")

print_section("✨ 所有测试完成")
print("\n🎯 下一步:")
print("   python train.py      # 开始训练")
print("   python generate.py   # 生成文本")
