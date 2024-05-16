# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such
import argparse
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model import GPTConfig, GPT

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

wind = 64
n_regist = 0

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model

# Parse command-line arguments for wind and n_regist
for arg in sys.argv[1:]:
    if arg.startswith('--wind='):
        wind = int(arg.split('=')[1])
    elif arg.startswith('--n_regist='):
        n_regist = int(arg.split('=')[1])

# Model configuration
mconf = GPTConfig(
    # ... other parameters ...
    wind=wind,
    n_regist=n_regist
)

# Initialize model
model = GPT(mconf)
