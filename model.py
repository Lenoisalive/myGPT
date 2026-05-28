# model.py
"""
V1: Bigram Language Model
最简单的语言模型 - 只考虑当前 token 预测下一个 token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Bigram 语言模型
    
    核心思想: P(next_token | current_token)
    每个 token 对应一组"下一个 token 的 logits"
    """
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Token Embedding Table: [vocab_size, vocab_size]
        # 本质: 每个 token 对应一组"下一个 token 的 logits"
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
        print(f"✅ 模型初始化完成")
        print(f"   词汇表大小: {vocab_size}")
        print(f"   参数量: {self.count_parameters():,}")
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def forward(self, idx, targets=None):
        """
        前向传播
        
        Args:
            idx: [B, T] 输入 token 序列
            targets: [B, T] 目标 token 序列 (可选)
            
        Returns:
            logits: [B, T, vocab_size] 每个位置对下一个 token 的预测
            loss: 如果提供了 targets，返回交叉熵损失
        """
        # idx: [B, T]
        # logits: [B, T, vocab_size]
        logits = self.token_embedding_table(idx)
        
        if targets is None:
            loss = None
        else:
            # 计算交叉熵损失
            # PyTorch 的 cross_entropy 需要: [B*T, C] 和 [B*T]
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)  # [B*T, vocab_size]
            targets_flat = targets.view(B*T)    # [B*T]
            loss = F.cross_entropy(logits_flat, targets_flat)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        生成文本
        
        Args:
            idx: [B, T] 初始上下文
            max_new_tokens: 生成的 token 数量
            temperature: 温度参数，控制随机性
                        - 1.0: 正常采样
                        - > 1.0: 更随机
                        - < 1.0: 更确定性
        
        Returns:
            [B, T+max_new_tokens] 生成的完整序列
        """
        for _ in range(max_new_tokens):
            # 获取预测
            # 注意: Bigram 只看最后一个 token
            logits, _ = self(idx)  # [B, T, vocab_size]
            
            # 只关注最后一个时间步
            logits = logits[:, -1, :]  # [B, vocab_size]
            
            # 应用温度
            logits = logits / temperature
            
            # 转换为概率
            probs = F.softmax(logits, dim=-1)  # [B, vocab_size]
            
            # 采样下一个 token
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]
            
            # 拼接到序列
            idx = torch.cat([idx, idx_next], dim=1)  # [B, T+1]
        
        return idx


def test_model():
    """测试模型基本功能"""
    print("\n" + "="*60)
    print("  🧪 测试 Bigram Language Model")
    print("="*60)
    
    # 模拟参数
    vocab_size = 65
    batch_size = 4
    block_size = 8
    
    print(f"\n📊 测试参数:")
    print(f"   vocab_size: {vocab_size}")
    print(f"   batch_size: {batch_size}")
    print(f"   block_size: {block_size}")
    
    # 创建模型
    print(f"\n🔨 创建模型...")
    model = BigramLanguageModel(vocab_size)
    
    # 创建随机输入
    print(f"\n📥 创建随机输入...")
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))
    
    print(f"   输入形状: {idx.shape}")
    print(f"   目标形状: {targets.shape}")
    
    # 前向传播
    print(f"\n⚡ 前向传播...")
    logits, loss = model(idx, targets)
    
    print(f"   输出形状: {logits.shape}  (期望: [{batch_size}, {block_size}, {vocab_size}])")
    print(f"   损失: {loss.item():.4f}")
    
    # 测试生成
    print(f"\n🎲 测试生成...")
    context = torch.zeros((1, 1), dtype=torch.long)  # 从 token 0 开始
    generated = model.generate(context, max_new_tokens=20)
    
    print(f"   生成序列形状: {generated.shape}")
    print(f"   生成的 tokens: {generated[0].tolist()}")
    
    print("\n" + "="*60)
    print("  ✅ 模型测试通过！")
    print("="*60)


if __name__ == "__main__":
    test_model()
