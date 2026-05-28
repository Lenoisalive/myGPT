# train.py
"""
训练语言模型
支持 V1 (Bigram)、V2 (Self-Attention)、V3 (Multi-Head Attention)、V4 (Transformer) 和 V5 (BPE)
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import time
import json
import os
import sys
from model import BigramLanguageModel
import config


# 从命令行参数获取版本，默认 V4
USE_ATTENTION = True
NUM_HEADS = config.n_head
N_LAYER = config.n_layer  # V4 使用多层
IS_V5 = False

if len(sys.argv) > 1:
    if sys.argv[1] == 'v1':
        USE_ATTENTION = False
        NUM_HEADS = 1
        N_LAYER = 0
    elif sys.argv[1] == 'v2':
        USE_ATTENTION = True
        NUM_HEADS = 1
        N_LAYER = 0
    elif sys.argv[1] == 'v3':
        USE_ATTENTION = True
        NUM_HEADS = config.n_head
        N_LAYER = 0
    elif sys.argv[1] == 'v4':
        USE_ATTENTION = True
        NUM_HEADS = config.n_head
        N_LAYER = config.n_layer
    elif sys.argv[1] == 'v5':
        USE_ATTENTION = True
        NUM_HEADS = config.n_head
        N_LAYER = config.n_layer
        IS_V5 = True

# 根据版本导入相应的数据集
if IS_V5:
    from dataset_v5 import get_batch, tokenizer, train_data, val_data
    print("✅ 使用 BPE Tokenizer (V5)")
else:
    from dataset import get_batch, tokenizer, train_data, val_data
    print("✅ 使用 Char Tokenizer (V1-V4)")


def print_section(title):
    """打印美化的分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


@torch.no_grad()
def estimate_loss(model, eval_iters=200):
    """
    评估模型在训练集和验证集上的损失
    
    Args:
        model: 模型
        eval_iters: 评估迭代次数
        
    Returns:
        dict: {'train': loss, 'val': loss}
    """
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    model.train()
    return out


def format_time(seconds):
    """格式化时间"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        return f"{seconds/60:.1f}分钟"
    else:
        return f"{seconds/3600:.1f}小时"


def train():
    """训练模型"""
    if not USE_ATTENTION:
        version_name = "V1 Bigram"
    elif IS_V5:
        version_name = f"V5 Transformer + BPE ({N_LAYER} layers, {NUM_HEADS} heads)"
    elif N_LAYER > 0:
        version_name = f"V4 Transformer ({N_LAYER} layers, {NUM_HEADS} heads)"
    elif NUM_HEADS == 1:
        version_name = "V2 Self-Attention"
    else:
        version_name = f"V3 Multi-Head Attention ({NUM_HEADS} heads)"
    
    print_section(f"🚀 开始训练 {version_name} Language Model")
    
    # 打印配置
    print("\n📋 训练配置:")
    print(f"   模型版本: {version_name}")
    print(f"   设备: {config.device}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   上下文长度: {config.block_size}")
    print(f"   最大迭代: {config.max_iters}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   评估间隔: {config.eval_interval}")
    if USE_ATTENTION:
        print(f"   嵌入维度: {config.n_embd}")
        if NUM_HEADS > 1:
            print(f"   Attention Heads: {NUM_HEADS}")
            print(f"   Head Size: {config.n_embd // NUM_HEADS}")
        if N_LAYER > 0:
            print(f"   Transformer Layers: {N_LAYER}")
            print(f"   Dropout: {config.dropout}")
    
    # 打印数据统计
    print(f"\n📊 数据统计:")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    print(f"   训练集大小: {len(train_data):,} tokens")
    print(f"   验证集大小: {len(val_data):,} tokens")
    
    # 创建模型
    print_section("🔨 创建模型")
    model = BigramLanguageModel(tokenizer.vocab_size, use_attention=USE_ATTENTION, num_heads=NUM_HEADS, n_layer=N_LAYER)
    model = model.to(config.device)
    
    # 创建优化器
    print(f"\n⚙️  创建优化器: AdamW")
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # 训练循环
    print_section("🏋️  开始训练循环")
    
    start_time = time.time()
    best_val_loss = float('inf')
    training_log = []  # 保存训练日志
    
    for iter in range(config.max_iters):
        # 定期评估
        if iter % config.eval_interval == 0 or iter == config.max_iters - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - start_time
            
            print(f"\n📈 步数 {iter:5d}/{config.max_iters}")
            print(f"   训练损失: {losses['train']:.4f}")
            print(f"   验证损失: {losses['val']:.4f}")
            print(f"   用时: {format_time(elapsed)}")
            
            # 记录日志
            training_log.append({
                'iter': iter,
                'train_loss': float(losses['train']),
                'val_loss': float(losses['val']),
                'elapsed_time': elapsed
            })
            
            # 保存最佳模型
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'iter': iter,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': losses['train'],
                    'val_loss': losses['val'],
                }, 'checkpoints/best_model.pt')
                print(f"   💾 保存最佳模型 (验证损失: {best_val_loss:.4f})")
            
            # 生成样本文本
            if iter % (config.eval_interval * 2) == 0:
                print(f"\n📝 生成样本文本:")
                model.eval()
                context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
                generated = model.generate(context, max_new_tokens=200, temperature=0.8)
                generated_text = tokenizer.decode(generated[0].tolist())
                print(f"\n{'─'*60}")
                print(generated_text)
                print(f"{'─'*60}")
                model.train()
        
        # 获取批次
        xb, yb = get_batch('train')
        
        # 前向传播
        logits, loss = model(xb, yb)
        
        # 反向传播
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        # 简单进度显示
        if iter % 100 == 0 and iter % config.eval_interval != 0:
            print(f".", end="", flush=True)
    
    # 训练完成
    total_time = time.time() - start_time
    if IS_V5:
        version_suffix = "v5"
    elif USE_ATTENTION:
        version_suffix = "v2"
    else:
        version_suffix = "v1"
    print_section("✨ 训练完成")
    print(f"   总用时: {format_time(total_time)}")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    
    # 保存训练日志
    log_filename = f'training_log_{version_suffix}.json'
    with open(log_filename, 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"   📊 训练日志已保存到 {log_filename}")
    
    # 保存最终模型
    torch.save({
        'iter': config.max_iters,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer,
    }, 'checkpoints/final_model.pt')
    print(f"   💾 最终模型已保存到 checkpoints/final_model.pt")
    
    # 最终生成
    print_section("🎉 最终生成样本")
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    
    for temp in [0.5, 0.8, 1.0]:
        print(f"\n🌡️  温度 = {temp}:")
        print(f"{'─'*60}")
        generated = model.generate(context, max_new_tokens=300, temperature=temp)
        generated_text = tokenizer.decode(generated[0].tolist())
        print(generated_text)
        print(f"{'─'*60}")
    
    return model


if __name__ == "__main__":
    # 确保 checkpoints 目录存在
    import os
    os.makedirs('checkpoints', exist_ok=True)
    
    # 开始训练
    model = train()
