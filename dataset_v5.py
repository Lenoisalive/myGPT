"""
V5 Dataset: 支持 BPE Tokenizer

与 dataset.py 的区别:
- 从 pickle 文件加载预处理的数据
- 支持 BPE tokenizer
- 更快的加载速度
"""

import torch
import pickle
import os
from tokenizer import BPETokenizer
import config


def print_section(title):
    """打印美化的分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


# 加载 BPE 数据
print_section("📖 加载 BPE 数据")
data_file = 'data/encoded_bpe.pkl'

if not os.path.exists(data_file):
    print(f"❌ 未找到 {data_file}")
    print(f"\n请先运行: python prepare_bpe_data.py")
    raise FileNotFoundError(f"请先运行 prepare_bpe_data.py 生成 BPE 数据")

print(f"⏳ 正在加载 {data_file}...")
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)

train_data = torch.tensor(data_dict['train'], dtype=torch.long)
val_data = torch.tensor(data_dict['val'], dtype=torch.long)
vocab_size = data_dict['vocab_size']
tokenizer_type = data_dict.get('tokenizer_type', 'bpe')
tokenizer_model = data_dict.get('tokenizer_model', 'gpt2')

print(f"✅ 数据加载完成")
print(f"📚 Tokenizer: {tokenizer_type} ({tokenizer_model})")
print(f"📚 词汇表大小: {vocab_size:,}")
print(f"🏋️  训练集: {len(train_data):,} tokens")
print(f"✅ 验证集: {len(val_data):,} tokens")

# 初始化 tokenizer (用于解码)
print_section("🔤 初始化 Tokenizer")
tokenizer = BPETokenizer(tokenizer_model)
assert tokenizer.vocab_size == vocab_size, f"词汇表大小不匹配: {tokenizer.vocab_size} vs {vocab_size}"
print(f"✅ Tokenizer 初始化完成")


def get_batch(split: str):
    """
    获取一个批次的训练数据
    
    Args:
        split: "train" 或 "val"
        
    Returns:
        x: 输入序列 [batch_size, block_size]
        y: 目标序列 [batch_size, block_size] (向右偏移一位)
    """
    data_source = train_data if split == "train" else val_data
    
    # 随机选择 batch_size 个起始位置
    ix = torch.randint(
        len(data_source) - config.block_size,
        (config.batch_size,)
    )
    
    # x: 当前 token 序列
    x = torch.stack([
        data_source[i:i + config.block_size]
        for i in ix
    ])
    
    # y: 向右偏移一位的目标序列
    y = torch.stack([
        data_source[i + 1:i + config.block_size + 1]
        for i in ix
    ])
    
    # 移动到指定设备
    x = x.to(config.device)
    y = y.to(config.device)
    
    return x, y


# 测试代码
if __name__ == "__main__":
    print_section("🧪 测试 get_batch 函数")
    
    print("⏳ 获取一个训练批次...")
    x, y = get_batch("train")
    print("✅ 批次获取完成\n")
    
    # 检查形状
    print("📐 张量形状:")
    print(f"  x.shape: {x.shape}  (期望: [{config.batch_size}, {config.block_size}])")
    print(f"  y.shape: {y.shape}  (期望: [{config.batch_size}, {config.block_size}])")
    print(f"  设备: {x.device}")
    
    # 显示第一个样本的 token ids
    print(f"\n🔢 第一个样本的 Token IDs (前10个):")
    print(f"  x[0][:10]: {x[0][:10].tolist()}")
    print(f"  y[0][:10]: {y[0][:10].tolist()}")
    
    # 解码并显示文本
    print(f"\n📖 解码第一个样本:")
    print(f"\n  X (输入序列):")
    x_text = tokenizer.decode(x[0].tolist())
    x_preview = x_text[:200] + ("..." if len(x_text) > 200 else "")
    print(f"  '{x_preview}'")
    
    print(f"\n  Y (目标序列，应该向右偏移一位):")
    y_text = tokenizer.decode(y[0].tolist())
    y_preview = y_text[:200] + ("..." if len(y_text) > 200 else "")
    print(f"  '{y_preview}'")
    
    print_section("✨ 测试完成")
    print(f"BPE Dataset 已准备就绪，可以开始训练 V5！")
