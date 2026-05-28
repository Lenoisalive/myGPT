# generate.py
"""
使用训练好的模型生成文本
"""

import torch
from model import BigramLanguageModel
from tokenizer import CharTokenizer
import config


def print_section(title):
    """打印美化的分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def load_model(checkpoint_path='checkpoints/best_model.pt'):
    """
    加载训练好的模型
    
    Args:
        checkpoint_path: 检查点路径
        
    Returns:
        model: 加载的模型
        tokenizer: tokenizer
    """
    print_section("📥 加载模型")
    print(f"⏳ 正在从 {checkpoint_path} 加载...")
    
    # 加载检查点
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    
    # 重建 tokenizer
    with open("data/input.txt", "r", encoding="utf-8") as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    
    # 创建模型
    model = BigramLanguageModel(tokenizer.vocab_size)
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


def interactive_mode():
    """交互式生成模式"""
    print_section("🎮 交互式生成模式")
    
    # 加载模型
    model, tokenizer = load_model()
    
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


def batch_generate():
    """批量生成多个样本"""
    print_section("🎲 批量生成样本")
    
    # 加载模型
    model, tokenizer = load_model()
    
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
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'batch':
            batch_generate()
        elif sys.argv[1] == 'interactive':
            interactive_mode()
        else:
            print("❌ 未知参数")
            print("用法:")
            print("  python generate.py              # 交互式模式")
            print("  python generate.py interactive  # 交互式模式")
            print("  python generate.py batch        # 批量生成")
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
