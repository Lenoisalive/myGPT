# generate.py
"""
使用训练好的模型生成文本
支持 V1、V2、V3、V4、V5、V6 和 V7 模型
"""

import torch
import sys
from model import BigramLanguageModel
from tokenizer import CharTokenizer, BPETokenizer
import config


def print_section(title):
    """打印美化的分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def load_model(checkpoint_path='checkpoints/best_model.pt', use_attention=True, num_heads=1, n_layer=0, is_v5=False, is_v6=False, is_v7=False):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点路径
        use_attention: 是否使用 attention (V2/V3/V4/V5/V6/V7)
        num_heads: attention heads 数量
        n_layer: transformer layers 数量
        is_v5: 是否是 V5 (BPE) 模型
        is_v6: 是否是 V6 (RoPE + RMSNorm + SwiGLU) 模型
        is_v7: 是否是 V7 (Better Training) 模型
        
    Returns:
        model: 加载的模型
        tokenizer: tokenizer
    """
    print_section("📥 加载模型")
    print(f"⏳ 正在从 {checkpoint_path} 加载...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 重建 tokenizer
    if is_v5 or is_v6 or is_v7:
        print("🔤 使用 BPE Tokenizer")
        tokenizer = BPETokenizer("gpt2")
    else:
        print("🔤 使用 Char Tokenizer")
        with open("data/input.txt", "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharTokenizer(text)
    
    # 创建模型 (V7 使用 V6 的架构)
    use_v6_arch = is_v6 or is_v7
    model = BigramLanguageModel(tokenizer.vocab_size, use_attention=use_attention, num_heads=num_heads, n_layer=n_layer, use_v6=use_v6_arch)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    model.eval()
    
    print(f"✅ 模型加载完成")
    if 'iter' in checkpoint:
        print(f"   训练步数: {checkpoint['iter']}")
    if 'val_loss' in checkpoint:
        print(f"   验证损失: {checkpoint['val_loss']:.4f}")
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt="", max_new_tokens=500, temperature=0.8):
    """
    生成文本
    
    Args:
        model: 模型
        tokenizer: tokenizer
        prompt: 初始提示文本
        max_new_tokens: 生成的 token 数量
        temperature: 温度参数
        
    Returns:
        generated_text: 生成的文本
    """
    model.eval()
    
    # 编码提示
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=config.device)
    else:
        # 从空白开始
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    
    # 生成
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature)
    
    # 解码
    generated_text = tokenizer.decode(generated[0].tolist())
    
    return generated_text


def interactive_mode(use_attention=True, num_heads=1, n_layer=0, is_v5=False, is_v6=False, is_v7=False):
    """交互式生成模式"""
    if not use_attention:
        version_name = "V1 (Bigram)"
    elif is_v7:
        version_name = f"V7 (Better Training, {n_layer} layers, {num_heads} heads) [Warmup + Cosine Decay]"
    elif is_v6:
        version_name = f"V6 (Modern Transformer, {n_layer} layers, {num_heads} heads) [RoPE + RMSNorm + SwiGLU]"
    elif is_v5:
        version_name = f"V5 (Transformer + BPE, {n_layer} layers, {num_heads} heads)"
    elif n_layer > 0:
        version_name = f"V4 (Transformer, {n_layer} layers, {num_heads} heads)"
    elif num_heads == 1:
        version_name = "V2 (Self-Attention)"
    else:
        version_name = f"V3 (Multi-Head Attention, {num_heads} heads)"
    
    print_section(f"🎮 交互式生成模式 - {version_name}")
    
    # 加载模型
    checkpoint_path = 'checkpoints/best_model.pt'
    model, tokenizer = load_model(checkpoint_path, use_attention=use_attention, num_heads=num_heads, n_layer=n_layer, is_v5=is_v5, is_v6=is_v6, is_v7=is_v7)
    
    print("\n💡 使用说明:")
    print("   - 输入提示文本，按回车生成")
    print("   - 输入 'quit' 或 'exit' 退出")
    print("   - 输入 'temp X' 设置温度 (例如: temp 0.8)")
    print("   - 直接按回车从空白生成")
    
    temperature = 0.8
    max_tokens = 300
    
    while True:
        print(f"\n{'='*60}")
        user_input = input(f"💬 提示 (温度={temperature}): ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("👋 再见!")
            break
        
        if user_input.lower().startswith('temp '):
            try:
                temperature = float(user_input.split()[1])
                print(f"✅ 温度已设置为 {temperature}")
                continue
            except:
                print("❌ 温度格式错误，请输入数字")
                continue
        
        print(f"\n⏳ 正在生成...")
        generated = generate_text(model, tokenizer, user_input, max_tokens, temperature)
        
        print(f"\n{'─'*60}")
        print(generated)
        print(f"{'─'*60}")


def batch_generate(use_attention=True, num_heads=1, n_layer=0, is_v5=False, is_v6=False, is_v7=False):
    """批量生成多个样本"""
    if not use_attention:
        version_name = "V1 (Bigram)"
    elif is_v7:
        version_name = f"V7 (Better Training, {n_layer} layers, {num_heads} heads) [Warmup + Cosine Decay]"
    elif is_v6:
        version_name = f"V6 (Modern Transformer, {n_layer} layers, {num_heads} heads) [RoPE + RMSNorm + SwiGLU]"
    elif is_v5:
        version_name = f"V5 (Transformer + BPE, {n_layer} layers, {num_heads} heads)"
    elif n_layer > 0:
        version_name = f"V4 (Transformer, {n_layer} layers, {num_heads} heads)"
    elif num_heads == 1:
        version_name = "V2 (Self-Attention)"
    else:
        version_name = f"V3 (Multi-Head Attention, {num_heads} heads)"
    
    print_section(f"🎲 批量生成样本 - {version_name}")
    
    # 加载模型
    model, tokenizer = load_model(use_attention=use_attention, num_heads=num_heads, n_layer=n_layer, is_v5=is_v5, is_v6=is_v6, is_v7=is_v7)
    
    temperatures = [0.5, 0.8, 1.0, 1.2]
    num_samples = 3
    
    for temp in temperatures:
        print(f"\n🌡️  温度 = {temp}")
        print("="*60)
        
        for i in range(num_samples):
            print(f"\n📝 样本 {i+1}:")
            print("─"*60)
            generated = generate_text(model, tokenizer, "", max_new_tokens=200, temperature=temp)
            print(generated)
            print("─"*60)


def main():
    """主函数"""
    import sys
    
    # 解析参数
    use_attention = True  # 默认使用 V4
    num_heads = config.n_head
    n_layer = config.n_layer
    is_v5 = False
    is_v6 = False
    is_v7 = False
    mode = 'interactive'
    
    for arg in sys.argv[1:]:
        if arg == 'v1':
            use_attention = False
            num_heads = 1
            n_layer = 0
            is_v5 = False
            is_v6 = False
            is_v7 = False
        elif arg == 'v2':
            use_attention = True
            num_heads = 1
            n_layer = 0
            is_v5 = False
            is_v6 = False
            is_v7 = False
        elif arg == 'v3':
            use_attention = True
            num_heads = config.n_head
            n_layer = 0
            is_v5 = False
            is_v6 = False
            is_v7 = False
        elif arg == 'v4':
            use_attention = True
            num_heads = config.n_head
            n_layer = config.n_layer
            is_v5 = False
            is_v6 = False
            is_v7 = False
        elif arg == 'v5':
            use_attention = True
            num_heads = config.n_head
            n_layer = config.n_layer
            is_v5 = True
            is_v6 = False
            is_v7 = False
        elif arg == 'v6':
            use_attention = True
            num_heads = config.n_head
            n_layer = config.n_layer
            is_v5 = False
            is_v6 = True
            is_v7 = False
        elif arg == 'v7':
            use_attention = True
            num_heads = config.n_head
            n_layer = config.n_layer
            is_v5 = False
            is_v6 = False
            is_v7 = True
        elif arg == 'batch':
            mode = 'batch'
        elif arg == 'interactive':
            mode = 'interactive'
    
    if mode == 'batch':
        batch_generate(use_attention, num_heads, n_layer, is_v5, is_v6, is_v7)
    else:
        interactive_mode(use_attention, num_heads, n_layer, is_v5, is_v6, is_v7)


if __name__ == "__main__":
    main()
