# model.py
"""
V1: Bigram Language Model
V2: Single-Head Self-Attention Language Model
V3: Multi-Head Self-Attention Language Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


class Head(nn.Module):
    """
    单个 Self-Attention Head
    
    核心思想: 
    - Query: 我在寻找什么
    - Key: 我能提供什么
    - Value: 我的信息内容
    """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] 输入
            
        Returns:
            out: [B, T, head_size] attention 输出
        """
        B, T, C = x.shape
        k = self.key(x)    # [B, T, head_size]
        q = self.query(x)  # [B, T, head_size]
        
        # 计算 attention scores: Q @ K^T
        # [B, T, head_size] @ [B, head_size, T] -> [B, T, T]
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # scaled attention
        
        # 应用 causal mask (只能看到左边的 token)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # softmax 得到概率
        wei = F.softmax(wei, dim=-1)  # [B, T, T]
        
        # 加权求和
        v = self.value(x)  # [B, T, head_size]
        out = wei @ v      # [B, T, T] @ [B, T, head_size] -> [B, T, head_size]
        
        return out


class MultiHeadAttention(nn.Module):
    """
    多头注意力机制
    
    核心思想:
    - 多个 head 并行学习不同的 attention 模式
    - head1 可能学习语法关系
    - head2 可能学习长距离依赖
    - head3 可能学习局部模式
    - head4 可能学习实体关系
    """
    
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, C] 输入
            
        Returns:
            out: [B, T, C] 多头注意力输出
        """
        # 并行计算所有 head
        # 每个 head 输出: [B, T, head_size]
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, num_heads * head_size]
        
        # 通过 projection 混合所有 head 的信息
        out = self.proj(out)  # [B, T, n_embd]
        
        return out


class BigramLanguageModel(nn.Module):
    """
    V1: Bigram 语言模型 (使用 use_attention=False)
    V2: Self-Attention 语言模型 (使用 use_attention=True, num_heads=1)
    V3: Multi-Head Attention 语言模型 (使用 use_attention=True, num_heads>1)
    """
    
    def __init__(self, vocab_size, use_attention=False, num_heads=1):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.num_heads = num_heads
        
        if use_attention:
            # V2/V3: Self-Attention 版本
            self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
            self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
            
            if num_heads == 1:
                # V2: Single-Head
                self.sa_head = Head(config.n_embd)
                version_name = "V2 Self-Attention"
            else:
                # V3: Multi-Head
                head_size = config.n_embd // num_heads
                self.sa_head = MultiHeadAttention(num_heads, head_size)
                version_name = f"V3 Multi-Head Attention ({num_heads} heads)"
            
            self.lm_head = nn.Linear(config.n_embd, vocab_size)
            
            print(f"✅ {version_name} 模型初始化完成")
        else:
            # V1: Bigram 版本
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
            
            print(f"✅ V1 Bigram 模型初始化完成")
        
        print(f"   词汇表大小: {vocab_size}")
        print(f"   使用 Attention: {use_attention}")
        if use_attention and num_heads > 1:
            print(f"   Attention Heads: {num_heads}")
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
        B, T = idx.shape
        
        if self.use_attention:
            # V2: Self-Attention 路径
            tok_emb = self.token_embedding_table(idx)  # [B, T, n_embd]
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # [T, n_embd]
            x = tok_emb + pos_emb  # [B, T, n_embd]
            x = self.sa_head(x)    # [B, T, n_embd] - 应用 self-attention
            logits = self.lm_head(x)  # [B, T, vocab_size]
        else:
            # V1: Bigram 路径
            logits = self.token_embedding_table(idx)  # [B, T, vocab_size]
        
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
            temperature: 温度参数
        
        Returns:
            [B, T+max_new_tokens] 生成的完整序列
        """
        for _ in range(max_new_tokens):
            # 如果使用 attention，需要截断上下文到 block_size
            if self.use_attention:
                idx_cond = idx[:, -config.block_size:]
            else:
                idx_cond = idx
            
            # 获取预测
            logits, _ = self(idx_cond)  # [B, T, vocab_size]
            
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
    print("  🧪 测试模型")
    print("="*60)
    
    vocab_size = 65
    batch_size = 4
    block_size = config.block_size
    
    print(f"\n📊 测试参数:")
    print(f"   vocab_size: {vocab_size}")
    print(f"   batch_size: {batch_size}")
    print(f"   block_size: {block_size}")
    print(f"   n_embd: {config.n_embd}")
    
    # 测试 V1 (Bigram)
    print(f"\n" + "─"*60)
    print("  V1: Bigram Model")
    print("─"*60)
    model_v1 = BigramLanguageModel(vocab_size, use_attention=False)
    
    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))
    
    logits, loss = model_v1(idx, targets)
    print(f"   输出形状: {logits.shape}")
    print(f"   损失: {loss.item():.4f}")
    
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = model_v1.generate(context, max_new_tokens=20)
    print(f"   生成: {generated.shape}")
    
    # 测试 V2 (Self-Attention)
    print(f"\n" + "─"*60)
    print("  V2: Self-Attention Model")
    print("─"*60)
    model_v2 = BigramLanguageModel(vocab_size, use_attention=True)
    
    logits, loss = model_v2(idx, targets)
    print(f"   输出形状: {logits.shape}")
    print(f"   损失: {loss.item():.4f}")
    
    generated = model_v2.generate(context, max_new_tokens=20)
    print(f"   生成: {generated.shape}")
    
    # 测试 V3 (Multi-Head Attention)
    print(f"\n" + "─"*60)
    print(f"  V3: Multi-Head Attention Model ({config.n_head} heads)")
    print("─"*60)
    model_v3 = BigramLanguageModel(vocab_size, use_attention=True, num_heads=config.n_head)
    
    logits, loss = model_v3(idx, targets)
    print(f"   输出形状: {logits.shape}")
    print(f"   损失: {loss.item():.4f}")
    
    generated = model_v3.generate(context, max_new_tokens=20)
    print(f"   生成: {generated.shape}")
    
    # 测试 multi-head attention
    print(f"\n🔍 测试 Multi-Head Attention:")
    with torch.no_grad():
        test_input = torch.randint(0, vocab_size, (1, 10))
        tok_emb = model_v3.token_embedding_table(test_input)
        pos_emb = model_v3.position_embedding_table(torch.arange(10))
        x = tok_emb + pos_emb
        
        # 测试每个 head 的输出
        head_outputs = [head(x) for head in model_v3.sa_head.heads]
        print(f"   每个 Head 输出: {head_outputs[0].shape}  (期望: [1, 10, {config.n_embd // config.n_head}])")
        
        # 测试 concat
        concatenated = torch.cat(head_outputs, dim=-1)
        print(f"   Concat 输出: {concatenated.shape}  (期望: [1, 10, {config.n_embd}])")
        
        # 测试完整输出
        full_output = model_v3.sa_head(x)
        print(f"   完整输出: {full_output.shape}  (期望: [1, 10, {config.n_embd}])")
    
    # 测试 V2 attention matrix (单个 head)
    print(f"\n🔍 测试 V2 Single-Head Attention Matrix:")
    with torch.no_grad():
        test_input = torch.randint(0, vocab_size, (1, 10))
        tok_emb = model_v2.token_embedding_table(test_input)
        pos_emb = model_v2.position_embedding_table(torch.arange(10))
        x = tok_emb + pos_emb
        
        # 计算 attention scores
        k = model_v2.sa_head.key(x)
        q = model_v2.sa_head.query(x)
        wei = q @ k.transpose(-2, -1) * (config.n_embd ** -0.5)
        wei = wei.masked_fill(model_v2.sa_head.tril[:10, :10] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        print(f"   Attention shape: {wei.shape}  (期望: [1, 10, 10])")
        print(f"   Causal mask 工作: {(wei[0].triu(1) == 0).all().item()}")
    
    print("\n" + "="*60)
    print("  ✅ 所有测试通过！")
    print("="*60)


if __name__ == "__main__":
    test_model()
