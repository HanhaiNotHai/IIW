import os
from time import strftime

WANDB = False
proxy: str | None = '127.0.0.1:7897'
if proxy is not None:
    os.environ['http_proxy'] = os.environ['https_proxy'] = proxy

epochs = 500
clamp = 2.0

# optimizer
lr = 1e-3

noise_flag = False

# input settings
stego_weight = 1
message_weight = 10
message_length = 64

# Train:
batch_size = 16
cropsize = 128
max_img2img_strength = 0.1

# Val:
cropsize_val = 128

# Data Path
TRAIN_PATH = 'dataset/DIV2K/DIV2K_train_HR_latents'
VAL_PATH = 'dataset/DIV2K/DIV2K_valid_HR_latents'

format_train = 'pt'
format_val = 'pt'

# Saving checkpoints:
MODEL_PATH = os.path.join('experiments', strftime('%m%d_%X'))
SAVE_freq = 1

suffix = ''
train_continue = False
diff = False

wandb_config = dict(
    epochs=epochs,
    lr=lr,
    message_weight=message_weight,
    stego_weight=stego_weight,
    batch_size=batch_size,
)
