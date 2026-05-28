# train.py
"""
训练 Bigram Language Model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
import time
import json
import os
from model import BigramLanguageModel
from dataset import get_batch, tokenizer, train_data, val_data
import config


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
    print_section("🚀 开始训练 Bigram Language Model")
    
    # 打印配置
    print("\n📋 训练配置:")
    print(f"   设备: {config.device}")
    print(f"   批次大小: {config.batch_size}")
    print(f"   上下文长度: {config.block_size}")
    print(f"   最大迭代: {config.max_iters}")
    print(f"   学习率: {config.learning_rate}")
    print(f"   评估间隔: {config.eval_interval}")
    
    # 打印数据统计
    print(f"\n📊 数据统计:")
    print(f"   词汇表大小: {tokenizer.vocab_size}")
    print(f"   训练集大小: {len(train_data):,} tokens")
    print(f"   验证集大小: {len(val_data):,} tokens")
    
    # 创建模型
    print_section("🔨 创建模型")
    model = BigramLanguageModel(tokenizer.vocab_size)
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
    print_section("✨ 训练完成")
    print(f"   总用时: {format_time(total_time)}")
    print(f"   最佳验证损失: {best_val_loss:.4f}")
    
    # 保存训练日志
    with open('training_log.json', 'w') as f:
        json.dump(training_log, f, indent=2)
    print(f"   📊 训练日志已保存到 training_log.json")
    
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
