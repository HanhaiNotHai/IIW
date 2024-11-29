import os
from time import strftime

WANDB = False
proxy: str | None = None  # '127.0.0.1:7897'
if proxy is not None:
    os.environ['http_proxy'] = os.environ['https_proxy'] = proxy

epochs = 500
clamp = 2.0

# optimizer
lr = 1e-3

noise_flag = False

# input settings
stego_weight = 2
message_weight = 10
constrained_weight = 0.1
message_length = 64

# Train:
batch_size = 40
cropsize = 128

# Val:
batchsize_val = 40
cropsize_val = 128

# Data Path
TRAIN_PATH = 'dataset/DIV2K/DIV2K_train_LR_x8'
VAL_PATH = 'dataset/DIV2K/DIV2K_valid_LR_x8'

format_train = 'png'
format_val = 'png'

# Saving checkpoints:
MODEL_PATH = os.path.join('experiments', strftime('%m%d_%X'))
SAVE_freq = 5

suffix = ''
train_continue = False
diff = False

wandb_config = dict(
    epochs=epochs,
    lr=lr,
    message_weight=message_weight,
    stego_weight=stego_weight,
    batch_size=batch_size,
    batchsize_val=batchsize_val,
)
