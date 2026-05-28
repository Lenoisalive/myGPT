# test_generate.py
"""
快速测试生成功能
"""

import torch
from model import BigramLanguageModel
from tokenizer import CharTokenizer
import config

print("="*60)
print("  🎮 测试文本生成")
print("="*60)

# 加载数据创建 tokenizer
print("\n⏳ 加载 tokenizer...")
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = CharTokenizer(text)
print(f"✅ 词汇表大小: {tokenizer.vocab_size}")

# 加载模型
print("\n⏳ 加载训练好的模型...")
model = BigramLanguageModel(tokenizer.vocab_size)

try:
    checkpoint = torch.load('checkpoints/best_model.pt', map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ 模型加载成功")
    print(f"   训练步数: {checkpoint['iter']}")
    print(f"   验证损失: {checkpoint['val_loss']:.4f}")
except FileNotFoundError:
    print("⚠️  未找到训练好的模型，使用随机初始化的模型")

model = model.to(config.device)
model.eval()

# 测试不同的提示和温度
print("\n" + "="*60)
print("  📝 生成样本")
print("="*60)

test_cases = [
    ("", 0.5, 150),
    ("", 0.8, 150),
    ("", 1.0, 150),
]

for i, (prompt, temp, length) in enumerate(test_cases, 1):
    print(f"\n{i}. 温度={temp}, 长度={length}")
    print("─"*60)
    
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=config.device)
        print(f"提示: '{prompt}'")
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        print(f"提示: (空白)")
    
    with torch.no_grad():
        generated = model.generate(context, max_new_tokens=length, temperature=temp)
    
    generated_text = tokenizer.decode(generated[0].tolist())
    print(generated_text)
    print("─"*60)

print("\n" + "="*60)
print("  ✅ 测试完成")
print("="*60)

print("\n💡 提示:")
print("   - 温度越低，生成越保守")
print("   - 温度越高，生成越随机")
print("   - 运行 'python generate.py' 进入交互模式")
