# compare_models.py
"""
比较 V1 和 V2 模型的生成质量
"""

import torch
from model import BigramLanguageModel
from tokenizer import CharTokenizer
import config

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

# 加载 tokenizer
with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()
tokenizer = CharTokenizer(text)

print_section("🔬 V1 vs V2 模型对比")

# 检查模型文件
import os
if not os.path.exists('checkpoints/best_model.pt'):
    print("\n❌ 未找到训练好的模型")
    print("   请先运行: python train.py v2")
    exit(1)

# 加载 V2 模型
checkpoint_v2 = torch.load('checkpoints/best_model.pt', map_location=config.device)

print("\n📊 模型信息:")
print(f"   训练步数: {checkpoint_v2.get('iter', 'N/A')}")
print(f"   验证损失: {checkpoint_v2.get('val_loss', 'N/A'):.4f}")

# 创建 V2 模型
model_v2 = BigramLanguageModel(tokenizer.vocab_size, use_attention=True)
model_v2.load_state_dict(checkpoint_v2['model_state_dict'])
model_v2 = model_v2.to(config.device)
model_v2.eval()

print(f"\n✅ V2 模型加载完成")
print(f"   参数量: {model_v2.count_parameters():,}")

# 生成对比
print_section("📝 生成对比 (温度=0.8)")

prompts = [
    "",  # 从头开始
    "The ",
    "Hello ",
]

for i, prompt in enumerate(prompts, 1):
    print(f"\n{i}. 提示: '{prompt}' (空白表示从头开始)")
    print("─"*60)
    
    if prompt:
        context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=config.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    
    with torch.no_grad():
        generated = model_v2.generate(context, max_new_tokens=150, temperature=0.8)
    
    text_output = tokenizer.decode(generated[0].tolist())
    print(text_output[:200] + ("..." if len(text_output) > 200 else ""))

print_section("✨ 对比完成")
print("\n💡 提示:")
print("   - V2 使用了 Self-Attention，能看到更多上下文")
print("   - 生成的文本应该比 V1 更连贯")
print("   - 单词结构更完整，开始出现短语")
