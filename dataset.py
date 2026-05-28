# dataset.py

import torch
import pickle
import os
from tokenizer import CharTokenizer, BPETokenizer
import config


def print_section(title):
    """打印美化的分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


# 1. 读取 input.txt
print_section("📖 加载数据")
print("⏳ 正在读取 data/input.txt...")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
print(f"✅ 读取完成: {len(text):,} 字符")

# 2. 用 tokenizer encode 成 token ids
print_section("🔤 创建 Tokenizer 并编码")
print("⏳ 正在初始化 tokenizer...")
tokenizer = CharTokenizer(text)
print(f"✅ Tokenizer 创建完成")
print(f"📚 词汇表大小: {tokenizer.vocab_size}")

print("\n⏳ 正在编码文本...")
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
print(f"✅ 编码完成: {len(data):,} tokens")

# 3. 划分 train / val (90% / 10%)
print_section("✂️  划分训练集和验证集")
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

print(f"📊 总数据量: {len(data):,} tokens")
print(f"🏋️  训练集: {len(train_data):,} tokens ({len(train_data)/len(data)*100:.1f}%)")
print(f"✅ 验证集: {len(val_data):,} tokens ({len(val_data)/len(data)*100:.1f}%)")
print(f"📦 批次大小: {config.batch_size}")
print(f"📏 上下文长度: {config.block_size}")


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
    
    # 验证 y 是否是 x 向右偏移一位
    print(f"\n✅ 验证偏移关系:")
    for i in range(min(5, config.block_size - 1)):
        match = "✓" if x[0][i+1].item() == y[0][i].item() else "✗"
        print(f"  {match} x[0][{i+1}] = {x[0][i+1].item()}  ==  y[0][{i}] = {y[0][i].item()}")
    
    # 解码并显示文本
    print(f"\n📖 解码第一个样本:")
    print(f"\n  X (输入序列):")
    x_text = tokenizer.decode(x[0].tolist())
    # 只显示前200个字符，避免输出过长
    x_preview = x_text[:200] + ("..." if len(x_text) > 200 else "")
    print(f"  '{x_preview}'")
    
    print(f"\n  Y (目标序列，应该向右偏移一位):")
    y_text = tokenizer.decode(y[0].tolist())
    y_preview = y_text[:200] + ("..." if len(y_text) > 200 else "")
    print(f"  '{y_preview}'")
    
    # 对比前几个字符
    print(f"\n🔍 详细对比 (前20个字符):")
    for i in range(min(20, len(x_text), len(y_text))):
        x_char = repr(x_text[i])[1:-1]  # 去掉引号
        y_char = repr(y_text[i])[1:-1]
        print(f"  位置 {i:2d}:  x[{i}]='{x_char}'  →  y[{i}]='{y_char}'")
    
    # 验证整体偏移
    print(f"\n🎯 整体验证:")
    # x 的后面部分应该等于 y 的前面部分
    match_count = (x[0][1:] == y[0][:-1]).sum().item()
    total = len(x[0]) - 1
    print(f"  匹配数量: {match_count}/{total}")
    if match_count == total:
        print(f"  ✅ 完美！y 确实是 x 向右偏移一位")
    else:
        print(f"  ❌ 错误！存在 {total - match_count} 个不匹配")
    
    print_section("✨ 测试完成")
    print(f"Dataset 已准备就绪，可以开始训练！")
