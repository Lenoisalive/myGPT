# config.py

batch_size = 32
block_size = 128        # context length
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4

n_embd = 128
n_head = 4              # V3 Multi-Head: number of attention heads
n_layer = 4
dropout = 0.2

device = "mps"  # 如果没有 GPU，改成 "cpu"

# V7: Better Training 配置
warmup_iters = 100      # Warmup 步数 (学习率线性增长)
lr_decay_iters = 5000   # 学习率衰减总步数
min_lr = 3e-5           # 最小学习率 (max_lr 的 1/10)
use_warmup = True       # 是否使用 warmup
use_cosine_decay = True # 是否使用 cosine decay