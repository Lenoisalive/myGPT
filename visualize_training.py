# visualize_training.py
"""
可视化训练过程
"""

import matplotlib.pyplot as plt
import json
import os


def plot_training_curves(log_file='training_log.json'):
    """
    绘制训练曲线
    
    Args:
        log_file: 日志文件路径
    """
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    # 读取日志
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    # 提取数据
    iters = [log['iter'] for log in logs]
    train_losses = [log['train_loss'] for log in logs]
    val_losses = [log['val_loss'] for log in logs]
    
    # 创建图表
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(iters, train_losses, label='训练损失', marker='o')
    plt.plot(iters, val_losses, label='验证损失', marker='s')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 过拟合检测
    plt.subplot(1, 2, 2)
    overfit = [val - train for train, val in zip(train_losses, val_losses)]
    plt.plot(iters, overfit, marker='o', color='red')
    plt.xlabel('迭代次数')
    plt.ylabel('验证损失 - 训练损失')
    plt.title('过拟合程度')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=150, bbox_inches='tight')
    print(f"✅ 训练曲线已保存到 training_curves.png")
    plt.show()


def print_summary(log_file='training_log.json'):
    """打印训练摘要"""
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    with open(log_file, 'r') as f:
        logs = json.load(f)
    
    print("\n" + "="*60)
    print("  📊 训练摘要")
    print("="*60)
    
    print(f"\n总迭代次数: {len(logs)}")
    print(f"起始损失: 训练={logs[0]['train_loss']:.4f}, 验证={logs[0]['val_loss']:.4f}")
    print(f"最终损失: 训练={logs[-1]['train_loss']:.4f}, 验证={logs[-1]['val_loss']:.4f}")
    
    # 找到最佳验证损失
    best_log = min(logs, key=lambda x: x['val_loss'])
    print(f"\n最佳验证损失: {best_log['val_loss']:.4f} (迭代 {best_log['iter']})")
    
    # 损失改善
    improvement = (logs[0]['val_loss'] - logs[-1]['val_loss']) / logs[0]['val_loss'] * 100
    print(f"损失改善: {improvement:.1f}%")
    
    print("="*60)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        log_file = sys.argv[1]
    else:
        log_file = 'training_log.json'
    
    print_summary(log_file)
    
    try:
        plot_training_curves(log_file)
    except ImportError:
        print("\n⚠️  matplotlib 未安装，无法绘制图表")
        print("   安装: pip install matplotlib")
