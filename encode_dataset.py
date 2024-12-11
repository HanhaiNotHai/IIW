import glob
import os
from itertools import batched

import torch
import torchvision.transforms.v2 as T
from natsort import natsorted
from torchvision.io import decode_image
from tqdm import tqdm

from models.vae import VAE

device = torch.device('cuda')
vae = VAE()
t = T.Compose(
    [
        T.Resize(1024),
        T.RandomCrop(1024),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.5], [0.5]),
    ]
)

for path in ['dataset/DIV2K/DIV2K_valid_HR', 'dataset/DIV2K/DIV2K_train_HR']:
    save_path = path + '_latents'
    os.makedirs(save_path, exist_ok=True)
    files = natsorted(sorted(glob.glob(path + "/*.png")))
    for file in tqdm(list(batched(files, 4))):
        x = torch.stack([t(decode_image(f)) for f in file])
        x = x.to(device)
        z = vae.encode_batch(x)
        z = z.to('cpu')
        for zi, f in zip(z, file):
            torch.save(zi, save_path + f[-9:-3] + 'pt')
