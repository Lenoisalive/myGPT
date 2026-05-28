#!/usr/bin/env python3
"""
从 HuggingFace FineWeb 数据集下载高质量文本数据
"""

import os
from datasets import load_dataset

def download_fineweb_data(target_size_mb=100, output_path="data/input.txt"):
    """
    从 FineWeb 下载数据
    
    Args:
        target_size_mb: 目标文件大小（MB），默认 100MB
        output_path: 输出文件路径
    """
    print("=" * 60)
    print("  📥 开始下载 FineWeb 数据集")
    print("=" * 60)
    print(f"\n📋 配置:")
    print(f"   目标大小: {target_size_mb} MB")
    print(f"   输出路径: {output_path}")
    
    # 确保 data 目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 备份旧文件
    if os.path.exists(output_path):
        backup_path = output_path + ".backup"
        print(f"\n💾 备份旧文件到: {backup_path}")
        os.rename(output_path, backup_path)
    
    # 加载数据集（流式）
    print("\n🌐 连接到 HuggingFace...")
    try:
        dataset = load_dataset(
            "HuggingFaceFW/fineweb",
            name="sample-10BT",
            split="train",
            streaming=True
        )
        print("✅ 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        print("\n💡 提示: 请确保已安装 datasets 库:")
        print("   pip install datasets")
        return False
    
    # 下载并写入文件
    target_size = target_size_mb * 1024 * 1024  # 转换为字节
    current_size = 0
    doc_count = 0
    
    print(f"\n📝 开始下载数据...")
    print("=" * 60)
    
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in dataset:
                text = item['text'] + "\n\n"
                f.write(text)
                current_size += len(text.encode('utf-8'))
                doc_count += 1
                
                # 每下载 10MB 显示进度
                if doc_count % 100 == 0:
                    progress = (current_size / target_size) * 100
                    mb_downloaded = current_size / (1024 * 1024)
                    print(f"📊 进度: {progress:.1f}% ({mb_downloaded:.1f} MB / {target_size_mb} MB, {doc_count} 文档)")
                
                if current_size >= target_size:
                    break
        
        print("=" * 60)
        print(f"\n✅ 下载完成！")
        print(f"   文件大小: {current_size / (1024 * 1024):.2f} MB")
        print(f"   文档数量: {doc_count}")
        print(f"   保存路径: {output_path}")
        
        # 显示文件统计
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            char_count = len(content)
            line_count = content.count('\n')
            word_count = len(content.split())
        
        print(f"\n📊 文件统计:")
        print(f"   字符数: {char_count:,}")
        print(f"   单词数: {word_count:,}")
        print(f"   行数: {line_count:,}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 下载过程中出错: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="从 FineWeb 下载训练数据")
    parser.add_argument(
        "--size",
        type=int,
        default=100,
        help="目标文件大小（MB），默认 100MB"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/input.txt",
        help="输出文件路径，默认 data/input.txt"
    )
    
    args = parser.parse_args()
    
    print("\n🚀 myGPT 数据下载工具")
    print("   下载高质量的 FineWeb 数据集\n")
    
    success = download_fineweb_data(args.size, args.output)
    
    if success:
        print("\n" + "=" * 60)
        print("  🎉 数据准备完成！")
        print("=" * 60)
        print("\n下一步:")
        print("  1. 运行 python train.py v2 开始训练")
        print("  2. 或查看数据: head -n 20 data/input.txt")
    else:
        print("\n❌ 下载失败，请检查错误信息")
