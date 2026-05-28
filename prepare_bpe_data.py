#!/usr/bin/env python3
"""
准备 BPE tokenized 数据

将原始文本使用 BPE tokenizer 编码，并保存为训练数据
"""

import os
import pickle
import numpy as np
from tokenizer import BPETokenizer


def prepare_bpe_data(input_file='data/input.txt', output_file='data/encoded_bpe.pkl'):
    """
    使用 BPE tokenizer 准备训练数据
    
    Args:
        input_file: 原始文本文件
        output_file: 输出的编码文件
    """
    print("=" * 70)
    print("🔧 准备 BPE 训练数据")
    print("=" * 70)
    
    # 读取文本
    print(f"\n📖 读取文本: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"   文本长度: {len(text):,} 字符")
    
    # 初始化 BPE tokenizer
    print(f"\n🔤 初始化 BPE Tokenizer...")
    tokenizer = BPETokenizer("gpt2")
    
    # 编码
    print(f"\n⚙️  编码文本...")
    data = tokenizer.encode(text)
    
    print(f"   Token 数量: {len(data):,}")
    print(f"   压缩比: {len(text)/len(data):.2f}x")
    print(f"   词汇表大小: {tokenizer.vocab_size:,}")
    
    # 转换为 numpy 数组
    data = np.array(data, dtype=np.int32)
    
    # 划分训练集和验证集
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    print(f"\n📊 数据划分:")
    print(f"   训练集: {len(train_data):,} tokens")
    print(f"   验证集: {len(val_data):,} tokens")
    
    # 保存
    print(f"\n💾 保存数据: {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'train': train_data,
            'val': val_data,
            'vocab_size': tokenizer.vocab_size,
            'tokenizer_type': 'bpe',
            'tokenizer_model': 'gpt2'
        }, f)
    
    print(f"\n✅ BPE 数据准备完成!")
    print(f"   文件大小: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # 显示统计信息
    print(f"\n📈 统计信息:")
    print(f"   训练 tokens: {len(train_data):,}")
    print(f"   验证 tokens: {len(val_data):,}")
    print(f"   词汇表: {tokenizer.vocab_size:,}")
    print(f"   平均 token/char: {len(data)/len(text):.2f}")
    
    # 显示示例
    print(f"\n🔍 示例:")
    sample_tokens = data[:50].tolist()
    sample_text = tokenizer.decode(sample_tokens)
    print(f"   前 50 tokens: {sample_tokens[:10]}...")
    print(f"   解码文本: {sample_text[:100]}...")
    
    return {
        'train_tokens': len(train_data),
        'val_tokens': len(val_data),
        'vocab_size': tokenizer.vocab_size,
        'compression_ratio': len(text)/len(data)
    }


def compare_with_char():
    """对比 Char vs BPE 数据"""
    print("\n" + "=" * 70)
    print("📊 Char vs BPE 对比")
    print("=" * 70)
    
    # 读取 char 数据
    try:
        with open('data/encoded.pkl', 'rb') as f:
            char_data = pickle.load(f)
        
        # 读取 BPE 数据
        with open('data/encoded_bpe.pkl', 'rb') as f:
            bpe_data = pickle.load(f)
        
        # 处理 char_data 的不同格式
        # 旧格式: {'encoded': array, 'tokenizer': obj, 'vocab_size': int}
        # 新格式: {'train': array, 'val': array, 'vocab_size': int}
        if 'encoded' in char_data:
            # 旧格式，需要手动划分
            char_encoded = char_data['encoded']
            n = len(char_encoded)
            char_train_len = int(n * 0.9)
            char_val_len = n - char_train_len
            char_vocab = char_data.get('vocab_size', 'N/A')
        else:
            # 新格式
            char_train_len = len(char_data['train'])
            char_val_len = len(char_data['val'])
            char_vocab = char_data.get('vocab_size', 'N/A')
        
        print(f"\n{'指标':<25} {'Char-level':<20} {'BPE':<20} {'改进':<15}")
        print(f"{'-'*80}")
        
        # 计算改进指标
        if char_vocab != 'N/A':
            vocab_improvement = bpe_data['vocab_size'] / char_vocab
            vocab_improvement_str = f"{vocab_improvement:.0f}x ⬆️"
        else:
            vocab_improvement_str = "N/A"
        
        train_reduction = char_train_len / len(bpe_data['train'])
        val_reduction = char_val_len / len(bpe_data['val'])
        
        print(f"{'词汇表大小':<25} {char_vocab:<20} {bpe_data['vocab_size']:<20} {vocab_improvement_str:<15}")
        print(f"{'训练 tokens':<25} {char_train_len:,:<20} {len(bpe_data['train']):,:<20} {f'{train_reduction:.1f}x ⬇️':<15}")
        print(f"{'验证 tokens':<25} {char_val_len:,:<20} {len(bpe_data['val']):,:<20} {f'{val_reduction:.1f}x ⬇️':<15}")
        
        # 文件大小
        char_size = os.path.getsize('data/encoded.pkl') / 1024
        bpe_size = os.path.getsize('data/encoded_bpe.pkl') / 1024
        size_reduction = char_size / bpe_size
        print(f"{'文件大小 (KB)':<25} {char_size:<20.1f} {bpe_size:<20.1f} {f'{size_reduction:.1f}x ⬇️':<15}")
        
        print(f"\n💡 结论:")
        reduction = (1 - len(bpe_data['train'])/char_train_len) * 100
        speedup = char_train_len / len(bpe_data['train'])
        print(f"   • BPE 将序列长度减少了 {reduction:.0f}%")
        print(f"   • 预计训练速度提升 {speedup:.1f}x")
        print(f"   • 预计显存占用减少 ~{reduction:.0f}%")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  无法对比: {e}")


if __name__ == "__main__":
    import sys
    
    # 准备 BPE 数据
    stats = prepare_bpe_data()
    
    # 对比
    compare_with_char()
    
    print("\n" + "=" * 70)
    print("🎯 下一步:")
    print("=" * 70)
    print("\n1️⃣ 训练 V5 模型 (使用 BPE):")
    print("   python train.py v5")
    print("\n2️⃣ 生成文本:")
    print("   python generate.py v5 --prompt 'Once upon a time'")
    print("\n" + "=" * 70)
