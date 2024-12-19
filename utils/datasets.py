import glob

import torch
import torchvision.transforms.v2 as T
from natsort import natsorted
from torch.utils.data import DataLoader, Dataset
from torchvision.io import decode_image

import config as c


class EncodeDataset(Dataset):
    def __init__(self, path, format):
        self.transform = T.Compose(
            [
                T.Resize(1024),
                T.RandomCrop(1024),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5], [0.5]),
            ]
        )

        self.files = natsorted(sorted(glob.glob(path + "/*." + format)))

    def __getitem__(self, index):
        try:
            image = decode_image(self.files[index])
            item = self.transform(image)
            return item
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


class DecodeDataset(Dataset):
    def __init__(self, path, format):
        self.transform = T.Compose(
            [
                T.Resize(1024),
                T.RandomCrop(1024),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize([0.5], [0.5]),
            ]
        )

        self.img1W = natsorted(sorted(glob.glob(path + "/*_1W." + format)))
        self.img2 = natsorted(sorted(glob.glob(path + "/*_2." + format)))
        self.img2W = natsorted(sorted(glob.glob(path + "/*_2W." + format)))

    def __getitem__(self, index):
        try:
            image1 = decode_image(self.img1W[index])
            item1 = self.transform(image1)
            image2 = decode_image(self.img2[index])
            item2 = self.transform(image2)
            image3 = decode_image(self.img2W[index])
            item3 = self.transform(image3)
            return item1, item2, item3
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.img1W)


class INN_Dataset(Dataset):
    def __init__(self, mode="train"):
        if mode == 'train':
            files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            files = natsorted(sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val)))
        self.items = [torch.load(f, weights_only=False) for f in files]

    def __getitem__(self, index):
        try:
            return self.items[index]
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.items)


def get_trainloader():
    trainloader = DataLoader(
        INN_Dataset(mode="train"),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader


def get_testloader():
    # Test Dataset loader
    testloader = DataLoader(
        INN_Dataset(mode="val"),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return testloader
