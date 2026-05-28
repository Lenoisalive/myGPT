import os
from tokenizer import CharTokenizer

def load_text(filepath):
    """加载文本文件"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def print_section_separator(title):
    """打印美化的分隔符"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def format_size(size_bytes):
    """格式化文件大小"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def main():
    # 文件路径
    input_file = 'data/input.txt'
    
    print_section_separator("加载数据")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"❌ 错误: 文件 {input_file} 不存在")
        return
    
    # 获取文件大小
    file_size = os.path.getsize(input_file)
    print(f"📁 文件路径: {input_file}")
    print(f"📊 文件大小: {format_size(file_size)}")
    
    # 加载文本
    print("\n⏳ 正在加载文本...")
    text = load_text(input_file)
    print(f"✅ 加载完成")
    
    # 文本统计
    print_section_separator("文本统计")
    print(f"📝 总字符数: {len(text):,}")
    print(f"📄 总行数: {text.count(chr(10)) + 1:,}")
    print(f"🔤 唯一字符数: {len(set(text))}")
    
    # 创建 tokenizer
    print_section_separator("创建 Tokenizer")
    print("⏳ 正在初始化 tokenizer...")
    tokenizer = CharTokenizer(text)
    print(f"✅ Tokenizer 创建完成")
    print(f"📚 词汇表大小: {tokenizer.vocab_size}")
    
    # 显示字符集预览
    print("\n🔤 字符集预览 (前50个):")
    preview_chars = tokenizer.chars[:50]
    # 将不可见字符转换为可见表示
    visible_chars = []
    for char in preview_chars:
        if char == '\n':
            visible_chars.append('\\n')
        elif char == '\t':
            visible_chars.append('\\t')
        elif char == ' ':
            visible_chars.append('␣')
        elif ord(char) < 32:
            visible_chars.append(f'\\x{ord(char):02x}')
        else:
            visible_chars.append(char)
    
    print('  ', ' '.join(visible_chars))
    if len(tokenizer.chars) > 50:
        print(f"  ... (还有 {len(tokenizer.chars) - 50} 个字符)")
    
    # 编码示例
    print_section_separator("编码示例")
    sample_text = text[:100] if len(text) >= 100 else text
    print(f"📖 原文 (前100字符):\n  {repr(sample_text)}")
    
    encoded = tokenizer.encode(sample_text)
    print(f"\n🔢 编码结果 (前20个token):")
    print(f"  {encoded[:20]}")
    if len(encoded) > 20:
        print(f"  ... (共 {len(encoded)} 个 tokens)")
    
    # 解码验证
    decoded = tokenizer.decode(encoded[:20])
    print(f"\n🔄 解码验证 (前20个token):")
    print(f"  {repr(decoded)}")
    
    # 完整编码统计
    print_section_separator("完整编码统计")
    print("⏳ 正在编码完整文本...")
    full_encoded = tokenizer.encode(text)
    print(f"✅ 编码完成")
    print(f"🎯 Token 总数: {len(full_encoded):,}")
    print(f"📊 压缩比: {len(full_encoded) / len(text):.2%} (1个字符 = 1个token)")
    
    # 保存 tokenizer
    print_section_separator("保存结果")
    
    # 可以选择保存编码后的数据
    save_choice = input("\n💾 是否保存编码后的数据? (y/n): ").lower()
    if save_choice == 'y':
        import pickle
        output_file = 'data/encoded.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump({
                'encoded': full_encoded,
                'tokenizer': tokenizer,
                'vocab_size': tokenizer.vocab_size
            }, f)
        print(f"✅ 已保存到 {output_file}")
    
    print_section_separator("处理完成")
    print("✨ 所有操作已完成!")

if __name__ == "__main__":
    main()
