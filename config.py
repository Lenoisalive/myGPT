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