import glob

import torchvision.transforms as T
from natsort import natsorted
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import config as c


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


class EncodeDataset(Dataset):
    def __init__(self, path, format):
        self.transform = T.Compose(
            [
                T.Resize([128, 128]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.files = natsorted(sorted(glob.glob(path + "/*." + format)))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
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
                T.Resize([128, 128]),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        self.files1 = natsorted(sorted(glob.glob(path + "/*_1." + format)))
        self.files2 = natsorted(sorted(glob.glob(path + "/*_2." + format)))

    def __getitem__(self, index):
        try:
            image1 = Image.open(self.files1[index])
            image1 = to_rgb(image1)
            item1 = self.transform(image1)
            image2 = Image.open(self.files2[index])
            image2 = to_rgb(image2)
            item2 = self.transform(image2)
            return item1, item2
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files1)


class INN_Dataset(Dataset):
    def __init__(self, transforms, mode="train"):
        self.transform = transforms
        self.mode = mode
        if mode == 'train':
            self.files = natsorted(sorted(glob.glob(c.TRAIN_PATH + "/*." + c.format_train)))
        else:
            self.files = natsorted(sorted(glob.glob(c.VAL_PATH + "/*." + c.format_val)))

    def __getitem__(self, index):
        try:
            image = Image.open(self.files[index])
            image = to_rgb(image)
            item = self.transform(image)
            return item
        except:
            return self.__getitem__(index + 1)

    def __len__(self):
        return len(self.files)


def get_trainloader():
    transform = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.Resize([c.cropsize, c.cropsize]),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    trainloader = DataLoader(
        INN_Dataset(transforms=transform, mode="train"),
        batch_size=c.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    return trainloader


def get_testloader():
    transform_val = T.Compose(
        [
            T.Resize([c.cropsize_val, c.cropsize_val]),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # Test Dataset loader
    testloader = DataLoader(
        INN_Dataset(transforms=transform_val, mode="val"),
        batch_size=c.batchsize_val,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
    )
    return testloader
