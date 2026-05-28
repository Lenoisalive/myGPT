#!/bin/bash
# 快速下载脚本 - 提供多种数据大小选项

echo "🚀 myGPT 数据下载快捷脚本"
echo ""
echo "请选择数据集大小："
echo "  1) 小型 - 50MB  (快速测试，~5分钟)"
echo "  2) 中型 - 100MB (推荐，~10分钟)"
echo "  3) 大型 - 500MB (更好效果，~30分钟)"
echo "  4) 超大 - 1GB   (最佳效果，~1小时)"
echo "  5) 自定义大小"
echo ""
read -p "请输入选项 (1-5): " choice

case $choice in
    1)
        size=50
        ;;
    2)
        size=100
        ;;
    3)
        size=500
        ;;
    4)
        size=1024
        ;;
    5)
        read -p "请输入自定义大小（MB）: " size
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "📥 开始下载 ${size}MB 数据..."
echo ""

# 激活虚拟环境并运行
source .venv/bin/activate
python download_data.py --size $size

echo ""
echo "✅ 完成！"
