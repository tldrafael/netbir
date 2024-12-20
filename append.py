import pandas as pd
import torch
import torchvision.transforms as T
import numpy as np
import cv2
import re
from PIL import Image
from torch.utils.data import Dataset
import lance
import random
import os


def get_newshape(oldh, oldw, long_length=1024):
    scale = long_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (neww, newh)


def resize_im(
        im, long_length=1024, fl_pad=False, inter_nearest=False
):

    target_size = get_newshape(*im.shape[:2], long_length=long_length)
    if im.ndim == 3 and im.shape[2] == 1:
        im = im[:, :, 0]

    if inter_nearest:
        inter = Image.Resampling.NEAREST
    else:
        inter = Image.Resampling.BICUBIC

    newim = Image.fromarray(im).resize(target_size, inter)
    newim = np.array(newim)

    if fl_pad:
        newh, neww = newim.shape[:2]
        padh = long_length - newh
        padw = long_length - neww

        if newim.ndim == 3 and newim.shape[2] == 3:
            pad_values = (124, 116, 104)
        else:
            pad_values = (0,)

        newim = cv2.copyMakeBorder(
            newim, 0, padh, 0, padw, cv2.BORDER_CONSTANT, value=pad_values)

        return newim, (padh, padw)
    else:
        return newim, None


def get_gtpath(p):
    p = re.sub('.jpg$', '.png', p)
    return p.replace('/im/', '/gt/')


def float_to_uint8(x):
    return (x * 255).round().clip(0, 255).astype(np.uint8)


def cv2_imread(p):
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)

    if im.ndim == 3:
        chs = im.shape[2]
        assert chs in [3, 4]

        if chs == 3:
            im = im[..., ::-1]
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2RGBA)

    return im


def bytes2im(s):
    s = np.asarray(bytearray(s), dtype=np.uint8)
    im = cv2.imdecode(s, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = im[..., ::-1]
    return im


@torch.no_grad()
def evaluate_evalset_by_cat(model, fl_fasttest=False, long=2048, flexai=False):

    if flexai:
        lancepath = os.path.join('/input/testcat.lance')
        dfcat = lance.dataset(lancepath)
        dfcat = dfcat.to_table().to_pandas()
    else:
        dfcat = pd.read_csv("/home/rafael/datasets/evalsets/evalset-multicat-v0.2-long2048/dfcat-for-training.csv")
        dfcat = dfcat.query('cat != "hanged-sod+prod"').reset_index(drop=True)

    dfcat['sad'] = np.nan

    if fl_fasttest:
        dfcat = dfcat.sample(2)

    T_pre = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x[None].cuda()),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])

    for i, r in dfcat.iterrows():
        if flexai:
            im = bytes2im(r.image)
        else:
            im = cv2_imread(r.path)
        oldh, oldw = im.shape[:2]

        newim, (padh, padw) = resize_im(
            im, long_length=long, fl_pad=True)

        input_ = T_pre(newim)
        pred = model(input_)[0]
        pred = pred.sigmoid()

        pred = pred[..., :(long-padh), :(long-padw)]
        pred = T.functional.resize(pred, (oldh, oldw), antialias=True)
        pred = pred[0, 0].cpu().numpy()

        if flexai:
            gt = bytes2im(r['mask']) / 255
        else:
            gt = cv2_imread(get_gtpath(r.path)) / 255
        sad = (pred - gt).__abs__().sum()
        pred = float_to_uint8(pred)
        dfcat.loc[i, 'sad'] = sad / 1000

    dfcat['sadlog'] = dfcat['sad'].apply(np.log2)
    return dfcat.groupby('cat')['sadlog'].mean().mean()


def read_trainlist(fpath):
    with open(fpath, 'r') as f:
        lines = f.read().split('\n')
    if len(lines[-1]) == 0:
        lines = lines[:-1]
    return lines


@torch.no_grad()
def evaluate_testglass(model, fl_fasttest=False):

    test_glass_3 = '/home/rafael/datasets/DSCollections/google-cars/cars-rgba-tidy-1-fillbg/valpaths.txt'
    ps = read_trainlist(test_glass_3)
    long = 1024

    if fl_fasttest:
        ps = ps[:2]

    T_pre = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x[None].cuda()),
        T.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])
    ])

    final = []
    for p in ps:
        im = cv2_imread(p)
        oldh, oldw = im.shape[:2]

        newim, (padh, padw) = resize_im(
            im, long_length=long, fl_pad=True)

        input_ = T_pre(newim)
        pred = model(input_)[0]
        pred = pred.sigmoid()

        # pred = pred[..., :(long-padh), :(long-padw)]
        # pred = T.functional.resize(pred, (oldh, oldw), antialias=True)
        pred = pred[0, 0].cpu().numpy()

        gt = cv2_imread(get_gtpath(p))
        newgt = resize_im(gt, long_length=long, fl_pad=True)[0] / 255
        sad = (pred - newgt).__abs__().sum() / 1000
        final.append(sad)

    return np.mean(final)


def set_randomseed(seed=None, return_seed=False):
    if seed is None:
        seed = np.random.randint(2147483647)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if return_seed:
        return seed


class Deit3Augs:

    def __init__(self):

        self.input = T.Compose([
            T.RandomHorizontalFlip(p=.5),
            T.RandomChoice([
                T.RandomGrayscale(p=1.),
                T.RandomSolarize(threshold=.5, p=1.),
                T.GaussianBlur(kernel_size=5)
            ]),
            T.ColorJitter(.3, .3, .3),
        ])
        self.target = T.Compose([
            T.RandomHorizontalFlip(p=.5),
        ])


class MyLanceDataset(Dataset):

    def __init__(self, lance_path, long=756):
        self.lanceds = lance.dataset(lance_path)
        self.long = long
        self.transforms = Deit3Augs()
        self.normalize = T.Normalize(
            mean=[.485, .456, .406], std=[.229, .224, .225]
        )

    def __len__(self):
        if hasattr(self.lanceds, 'count_rows'):
            return self.lanceds.count_rows()
        else:
            return self.lanceds.num_rows

    def load_data(self, idx):
        raw = self.lanceds.take([idx], columns=['image', 'mask']).to_pydict()

        im = np.frombuffer(b''.join(raw['image']), dtype=np.uint8)
        im = cv2.imdecode(im, cv2.IMREAD_UNCHANGED)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        im = resize_im(im, self.long, fl_pad=True)[0]

        gt = np.frombuffer(b''.join(raw['mask']), dtype=np.uint8)
        gt = cv2.imdecode(gt, cv2.IMREAD_UNCHANGED)
        gt = resize_im(gt, self.long, fl_pad=True)[0]

        return im, gt

    def __getitem__(self, idx):
        im, gt = self.load_data(idx)

        im = T.functional.to_tensor(im)
        gt = T.functional.to_tensor(gt)

        if self.transforms:
            state = set_randomseed(return_seed=True)
            im = self.transforms.input(im)

        if gt.ndim == 3 and gt.shape[0] > 1:
            gt = gt[[0]]

        if self.transforms and self.transforms.target:
            set_randomseed(seed=state)
            gt = self.transforms.target(gt)
            set_randomseed(seed=state)

        if im.shape[0] == 1:
            im = torch.cat([im for _ in range(3)], axis=0)

        im = self.normalize(im)

        class_label = -1
        return im, gt, class_label
