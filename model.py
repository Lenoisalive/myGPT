# model.py
"""
V1: Bigram Language Model
V2: Single-Head Self-Attention Language Model
V3: Multi-Head Self-Attention Language Model
V4: Transformer Block (with FFN, Residual, LayerNorm)
V5: BPE Tokenizer + Transformer
V6: RoPE + RMSNorm + SwiGLU (Modern Architecture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import math
import config


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    优势相比 LayerNorm:
    - 更简单：不需要计算均值和方差
    - 更快：计算量更少
    - 更稳定：训练更稳定
    - Llama/Mistral 等现代模型都使用 RMSNorm
    
    公式: y = x / RMS(x) * scale
    其中 RMS(x) = sqrt(mean(x^2) + eps)
    """
    
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # x: [B, T, C]
        # 计算 RMS
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        # 归一化并缩放
        x_normed = x / rms
        return self.scale * x_normed


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    
    优势相比传统 Position Embedding:
    - 相对位置编码：直接编码相对位置信息
    - 外推能力强：可以推广到训练时未见过的序列长度
    - 参数更少：不需要可学习的位置 embedding 表
    - GPT-NeoX、Llama、PaLM 等都使用 RoPE
    
    核心思想：
    - 将 query 和 key 向量按维度分组
    - 每组应用旋转矩阵（rotation matrix）
    - 旋转角度随位置变化
    - 使得 attention score 包含相对位置信息
    """
    
    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # 预计算旋转频率
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算 cos 和 sin 值
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # [max_seq_len, dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_seq_len, dim]
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
    
    def rotate_half(self, x):
        """将输入的后半部分旋转"""
        x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x, seq_len=None):
        """
        应用 RoPE
        
        Args:
            x: [B, num_heads, T, head_dim] 或 [B, T, dim]
            seq_len: 序列长度（可选）
        
        Returns:
            [B, num_heads, T, head_dim] 或 [B, T, dim] 应用 RoPE 后的结果
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[:seq_len, ...]
        sin = self.sin_cached[:seq_len, ...]
        
        # 广播到正确的形状
        if len(x.shape) == 4:  # [B, num_heads, T, head_dim]
            cos = cos[None, None, :, :]  # [1, 1, T, head_dim]
            sin = sin[None, None, :, :]
        else:  # [B, T, dim]
            cos = cos[None, :, :]  # [1, T, dim]
            sin = sin[None, :, :]
        
        # 应用旋转
        return (x * cos) + (self.rotate_half(x) * sin)


class SwiGLU(nn.Module):
    """
    Swish-Gated Linear Unit (SwiGLU)
    
    优势相比标准 FFN:
    - 表达能力更强：门控机制可以选择性地传递信息
    - 性能更好：在相同参数量下表现更好
    - Llama、PaLM 等现代模型都使用 SwiGLU
    
    公式: SwiGLU(x) = Swish(xW) ⊙ (xV)
    其中 Swish(x) = x * sigmoid(x)
         ⊙ 表示逐元素乘法
    
    相比标准 FFN:
    标准: Linear(GELU(Linear(x)))
    SwiGLU: Linear(Swish(Linear(x)) * Linear(x))
    """
    
    def __init__(self, n_embd, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            # SwiGLU 通常使用 8/3 倍扩展（约2.67倍）而不是 4 倍
            # 这样在相同计算量下参数更多
            hidden_dim = int(8 * n_embd / 3)
            # 确保是 256 的倍数（优化内存对齐）
            hidden_dim = ((hidden_dim + 255) // 256) * 256
        
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # Gate
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)  # Down projection
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        # x: [B, T, n_embd]
        # SwiGLU: swish(xW1) * (xW3) then project down with W2
        swish_gate = F.silu(self.w1(x))  # [B, T, hidden_dim], silu = swish
        x_up = self.w3(x)                 # [B, T, hidden_dim]
        x = swish_gate * x_up             # [B, T, hidden_dim]
        x = self.w2(x)                    # [B, T, n_embd]
        x = self.dropout(x)
        return x


class Head(nn.Module):
    """
    单个 Self-Attention Head
    
    核心思想: 
    - Query: 我在寻找什么
    - Key: 我能提供什么
    - Value: 我的信息内容
    
    V6 改进:
    - 支持 RoPE 位置编码
    """
    
    def __init__(self, head_size, use_rope=False):
        super().__init__()
        self.head_size = head_size
        self.use_rope = use_rope
        
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        
        if use_rope:
            self.rope = RoPE(head_size, max_seq_len=config.block_size)
        
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
        
        # 应用 RoPE（如果启用）
        if self.use_rope:
            q = self.rope(q, seq_len=T)
            k = self.rope(k, seq_len=T)
        
        # 计算 attention scores: Q @ K^T
        # [B, T, head_size] @ [B, head_size, T] -> [B, T, T]
        wei = q @ k.transpose(-2, -1) * (self.head_size ** -0.5)  # scaled attention
        
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
    
    V6 改进:
    - 支持 RoPE 位置编码
    """
    
    def __init__(self, num_heads, head_size, use_rope=False):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, use_rope=use_rope) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
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
        out = self.dropout(out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed-Forward 网络
    
    核心思想:
    - 简单的两层 MLP
    - 中间层扩展 4 倍（GPT 标准做法）
    - GELU 激活函数（比 ReLU 更平滑）
    - Dropout 防止过拟合
    """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # 扩展 4 倍
            nn.GELU(),                       # GPT 使用 GELU
            nn.Linear(4 * n_embd, n_embd),  # 投影回原维度
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    完整的 Transformer Block
    
    V4 结构:
        x → LayerNorm → MultiHeadAttention → Residual
          → LayerNorm → FeedForward → Residual
    
    V6 结构 (use_v6=True):
        x → RMSNorm → MultiHeadAttention(RoPE) → Residual
          → RMSNorm → SwiGLU → Residual
    
    核心组件:
    - V4: LayerNorm, FFN
    - V6: RMSNorm, SwiGLU, RoPE
    - Residual: 避免梯度消失
    - Attention: 信息交互
    """
    
    def __init__(self, n_embd, num_heads, use_v6=False):
        super().__init__()
        head_size = n_embd // num_heads
        self.use_v6 = use_v6
        
        # Attention
        self.sa = MultiHeadAttention(num_heads, head_size, use_rope=use_v6)
        
        # FFN / SwiGLU
        if use_v6:
            self.ffwd = SwiGLU(n_embd)
        else:
            self.ffwd = FeedForward(n_embd)
        
        # Normalization
        if use_v6:
            self.ln1 = RMSNorm(n_embd)
            self.ln2 = RMSNorm(n_embd)
        else:
            self.ln1 = nn.LayerNorm(n_embd)
            self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        """
        Args:
            x: [B, T, C] 输入
            
        Returns:
            out: [B, T, C] 输出
        """
        # Attention block with residual
        x = x + self.sa(self.ln1(x))    # Pre-norm + residual
        
        # Feed-forward block with residual
        x = x + self.ffwd(self.ln2(x))  # Pre-norm + residual
        
        return x


class BigramLanguageModel(nn.Module):
    """
    V1: Bigram 语言模型 (use_attention=False)
    V2: Self-Attention 语言模型 (use_attention=True, num_heads=1, n_layer=0)
    V3: Multi-Head Attention 语言模型 (use_attention=True, num_heads>1, n_layer=0)
    V4: Transformer Block 语言模型 (use_attention=True, num_heads>1, n_layer>0)
    V5: BPE + Transformer (use_attention=True, num_heads>1, n_layer>0)
    V6: RoPE + RMSNorm + SwiGLU (use_attention=True, num_heads>1, n_layer>0, use_v6=True)
    """
    
    def __init__(self, vocab_size, use_attention=False, num_heads=1, n_layer=0, use_v6=False):
        super().__init__()
        self.vocab_size = vocab_size
        self.use_attention = use_attention
        self.num_heads = num_heads
        self.n_layer = n_layer
        self.use_v6 = use_v6
        
        if use_attention:
            # V2/V3/V4/V5/V6: Attention-based 版本
            self.token_embedding_table = nn.Embedding(vocab_size, config.n_embd)
            
            # V6 使用 RoPE，不需要 position embedding
            if not use_v6:
                self.position_embedding_table = nn.Embedding(config.block_size, config.n_embd)
            
            if n_layer > 0:
                # V4/V5/V6: Transformer Blocks
                self.blocks = nn.Sequential(*[
                    TransformerBlock(config.n_embd, num_heads, use_v6=use_v6) 
                    for _ in range(n_layer)
                ])
                
                # Final layer norm
                if use_v6:
                    self.ln_f = RMSNorm(config.n_embd)
                    version_name = f"V6 Modern Transformer ({n_layer} layers, {num_heads} heads) [RoPE + RMSNorm + SwiGLU]"
                else:
                    self.ln_f = nn.LayerNorm(config.n_embd)
                    version_name = f"V4 Transformer ({n_layer} layers, {num_heads} heads)"
            elif num_heads == 1:
                # V2: Single-Head
                self.sa_head = Head(config.n_embd, use_rope=use_v6)
                version_name = "V2 Self-Attention"
            else:
                # V3: Multi-Head
                head_size = config.n_embd // num_heads
                self.sa_head = MultiHeadAttention(num_heads, head_size, use_rope=use_v6)
                version_name = f"V3 Multi-Head Attention ({num_heads} heads)"
            
            self.lm_head = nn.Linear(config.n_embd, vocab_size)
            
            print(f"✅ {version_name} 模型初始化完成")
        else:
            # V1: Bigram 版本
            self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
            
            print(f"✅ V1 Bigram 模型初始化完成")
        
        print(f"   词汇表大小: {vocab_size}")
        print(f"   使用 Attention: {use_attention}")
        if use_attention:
            if n_layer > 0:
                print(f"   Transformer Layers: {n_layer}")
                if use_v6:
                    print(f"   架构改进: RoPE + RMSNorm + SwiGLU")
            if num_heads > 1:
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
            # V2/V3/V4/V5/V6: Attention 路径
            tok_emb = self.token_embedding_table(idx)  # [B, T, n_embd]
            
            # V6 使用 RoPE，不需要 position embedding
            if self.use_v6:
                x = tok_emb  # [B, T, n_embd]
            else:
                pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))  # [T, n_embd]
                x = tok_emb + pos_emb  # [B, T, n_embd]
            
            if self.n_layer > 0:
                # V4/V5/V6: 通过 Transformer blocks
                x = self.blocks(x)  # [B, T, n_embd]
                x = self.ln_f(x)    # Final layer norm
            else:
                # V2/V3: 单层 attention
                x = self.sa_head(x)  # [B, T, n_embd]
            
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
    
    # 测试 V4 (Transformer Blocks)
    print(f"\n" + "─"*60)
    print(f"  V4: Transformer Model ({config.n_layer} layers, {config.n_head} heads)")
    print("─"*60)
    model_v4 = BigramLanguageModel(vocab_size, use_attention=True, num_heads=config.n_head, n_layer=config.n_layer)
    
    logits, loss = model_v4(idx, targets)
    print(f"   输出形状: {logits.shape}")
    print(f"   损失: {loss.item():.4f}")
    
    generated = model_v4.generate(context, max_new_tokens=20)
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
