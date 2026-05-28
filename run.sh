#!/bin/bash
# run.sh - 快速启动脚本

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
PROJECT_DIR="/Users/sulingjie/projects/myGPT"
PYTHON="${PROJECT_DIR}/.venv/bin/python"

# 打印带颜色的消息
print_message() {
    color=$1
    message=$2
    echo -e "${color}${message}${NC}"
}

# 显示菜单
show_menu() {
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "           🤖 myGPT - V1 Bigram Language Model"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "请选择操作:"
    echo ""
    echo "  1) 🧪 快速测试所有组件"
    echo "  2) 📊 查看数据统计"
    echo "  3) 🏋️  开始训练模型"
    echo "  4) 📝 交互式生成文本"
    echo "  5) 🎲 批量生成样本"
    echo "  6) 📈 可视化训练曲线"
    echo "  7) 🔍 测试生成功能"
    echo "  8) 📚 查看文档"
    echo "  9) ℹ️  项目信息"
    echo "  0) 🚪 退出"
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo -n "请输入选项 [0-9]: "
}

# 按任意键继续
press_any_key() {
    echo ""
    read -n 1 -s -r -p "按任意键继续..."
    echo ""
}

# 1. 快速测试
run_quick_test() {
    print_message "$BLUE" "🧪 运行快速测试..."
    cd "$PROJECT_DIR"
    $PYTHON quick_test.py
    press_any_key
}

# 2. 数据统计
show_data_stats() {
    print_message "$BLUE" "📊 显示数据统计..."
    cd "$PROJECT_DIR"
    $PYTHON dataset.py
    press_any_key
}

# 3. 训练模型
train_model() {
    print_message "$YELLOW" "⚠️  注意: 训练将覆盖现有模型！"
    read -p "确定要开始训练吗? (y/n): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        print_message "$BLUE" "🏋️  开始训练..."
        cd "$PROJECT_DIR"
        $PYTHON train.py
    else
        print_message "$GREEN" "已取消训练"
    fi
    press_any_key
}

# 4. 交互式生成
interactive_generate() {
    print_message "$BLUE" "📝 启动交互式生成..."
    cd "$PROJECT_DIR"
    $PYTHON generate.py interactive
    press_any_key
}

# 5. 批量生成
batch_generate() {
    print_message "$BLUE" "🎲 批量生成样本..."
    cd "$PROJECT_DIR"
    $PYTHON generate.py batch
    press_any_key
}

# 6. 可视化
visualize() {
    print_message "$BLUE" "📈 生成训练曲线..."
    cd "$PROJECT_DIR"
    if [ -f "training_log.json" ]; then
        $PYTHON visualize_training.py
    else
        print_message "$RED" "❌ 未找到训练日志文件"
        print_message "$YELLOW" "请先运行训练"
    fi
    press_any_key
}

# 7. 测试生成
test_generate() {
    print_message "$BLUE" "🔍 测试生成功能..."
    cd "$PROJECT_DIR"
    $PYTHON test_generate.py
    press_any_key
}

# 8. 查看文档
show_docs() {
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "                      📚 项目文档"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "主要文档:"
    echo "  1) README.md            - 项目主页"
    echo "  2) GUIDE.md             - 完整使用指南"
    echo "  3) TRAINING_SUMMARY.md  - 训练总结"
    echo "  4) PROJECT_COMPLETE.md  - 完成报告"
    echo "  5) FINAL_SUMMARY.md     - 最终总结"
    echo ""
    echo -n "输入文档编号查看 (1-5) 或按回车返回: "
    read doc_choice
    
    case $doc_choice in
        1) less README.md ;;
        2) less GUIDE.md ;;
        3) less TRAINING_SUMMARY.md ;;
        4) less PROJECT_COMPLETE.md ;;
        5) less FINAL_SUMMARY.md ;;
        *) return ;;
    esac
}

# 9. 项目信息
show_info() {
    clear
    echo "════════════════════════════════════════════════════════════"
    echo "                   ℹ️  项目信息"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "📦 项目名称:  myGPT"
    echo "📌 当前版本:  V1 - Bigram Language Model"
    echo "📅 完成日期:  2026年5月28日"
    echo "💻 Python:    $($PYTHON --version)"
    echo "🔥 PyTorch:   $($PYTHON -c 'import torch; print(torch.__version__)')"
    echo ""
    echo "📊 模型信息:"
    if [ -f "checkpoints/best_model.pt" ]; then
        echo "  ✅ 最佳模型已训练"
        echo "  📁 模型路径: checkpoints/best_model.pt"
        model_size=$(du -h checkpoints/best_model.pt | cut -f1)
        echo "  💾 模型大小: $model_size"
    else
        echo "  ⚠️  未找到训练好的模型"
    fi
    echo ""
    echo "📈 训练日志:"
    if [ -f "training_log.json" ]; then
        echo "  ✅ 训练日志存在"
        log_lines=$(wc -l < training_log.json)
        echo "  📝 日志行数: $log_lines"
    else
        echo "  ⚠️  未找到训练日志"
    fi
    echo ""
    echo "📁 项目结构:"
    echo "  📄 核心代码:  7 个文件"
    echo "  🧪 测试工具:  4 个文件"
    echo "  📚 文档:      5 个文件"
    echo ""
    press_any_key
}

# 主循环
main() {
    while true; do
        show_menu
        read choice
        
        case $choice in
            1) run_quick_test ;;
            2) show_data_stats ;;
            3) train_model ;;
            4) interactive_generate ;;
            5) batch_generate ;;
            6) visualize ;;
            7) test_generate ;;
            8) show_docs ;;
            9) show_info ;;
            0) 
                print_message "$GREEN" "👋 再见!"
                exit 0
                ;;
            *)
                print_message "$RED" "❌ 无效选项，请重新选择"
                sleep 1
                ;;
        esac
    done
}

# 运行主程序
main
