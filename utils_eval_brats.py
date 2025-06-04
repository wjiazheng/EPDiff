
"""
This code is based on @dome272 implementation of DDPM's
https://github.com/dome272/Diffusion-Models-pytorch
"""

import os
import random
import glob
import nibabel as nib
import numpy as np
import pandas as pd
import skimage.exposure as ex
import torch
import torch.nn as nn
import torchvision
from matplotlib import pyplot as plt
from PIL import Image
from scipy.ndimage import median_filter
from scipy.ndimage import grey_dilation
from scipy.ndimage import binary_dilation
from scipy.signal import medfilt2d
from skimage.measure import label, regionprops
from torch.nn import functional as F
import torchvision.transforms.functional as tf
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from skimage import transform
from concurrent.futures import ThreadPoolExecutor


def plot_images(images, mode="RGB"):
    if mode == "L": # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu(),
            cmap="gray",
        )
        plt.show()
    else:
        plt.figure(figsize=(32, 32))
        plt.imshow(
            torch.cat(
                [
                    torch.cat([i for i in images.cpu()], dim=-1),
                ],
                dim=-2,
            )
            .permute(1, 2, 0)
            .cpu()
        )
        plt.show()


def save_images(images, path, mode="RGB", **kwargs):
    if mode == "L":  # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
        im = Image.fromarray(ndarr)
        im.save(path)


def upload_images(images, mode="RGB", **kwargs):
    """Creates a numpy array to upload the images to wandb.

    Parameters
    ----------
    images : tensor
        The tensor containing the images
    mode : str, optional
        flag that decides if the images are meant to be RBG or gray scale, by default "RGB"

    Returns
    -------
    numpy.array
        The array that can be uploaded to wandb
    """
    if mode == "L": # mode L is gray scale
        batch_size = images.shape[0]
        channels = images.shape[1]
        images = images.reshape(
            (batch_size * channels, 1, images.shape[2], images.shape[3])
        )
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    else:
        grid = torchvision.utils.make_grid(images, **kwargs)
        ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    return ndarr


class MRIDataVolume(Dataset):
    """The data set class to load and normalize complete volumes.

    Parameters
    ----------
    Dataset : _type_
        PyTorch class
    """
    def __init__(
        self,
        df: pd.DataFrame,
        dataset_path: str,
        image_size: int,
        hist: bool,
    ):
        self.df = df
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.hist = hist
        self.data_types = ["_flair.nii.gz", "_t1.nii.gz", "_t1ce.nii.gz", "_t2.nii.gz"]

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        id_ = self.df.loc[idx, self.df.columns[0]]
        patient_number = id_
        images = []
        for data_type in self.data_types:
            img_path = os.path.join(self.dataset_path, id_, patient_number + data_type)
            img = np.asarray(nib.load(img_path).dataobj, dtype=float)
            img = torch.from_numpy(img)
            images.append(img)

        mask_path = os.path.join(self.dataset_path, id_, patient_number + "_seg.nii.gz")
        mask = np.asarray(nib.load(mask_path).dataobj, dtype=float)
        mask[mask > 0.5] = 1  # for data sets where the gt masks need to be registered a thr needs to be decided
        mask[mask < 1] = 0
        mask = torch.from_numpy(mask)
        mask = F.interpolate(mask[None, None], [self.image_size, self.image_size, 155], mode="nearest-exact")
        mask = mask[0, 0].type(torch.bool)
        if self.hist == True:
            img = np.stack([x for x in images])
            img = hist_norm(img)
        else:
            img = torch.stack([x for x in images], dim=0)
            img = normalize_volume(img.float())

        volume = torch.zeros(
            img.shape[0], self.image_size, self.image_size, mask.shape[2]
        )
        my_transform = transforms.Resize(self.image_size, antialias=True)
        for i in range(mask.shape[2]):
            volume[:, :, :, i] = my_transform(img[None, :, :, :, i])

        return volume, mask

def MRI_Volume(conf, hist=False):
    df = pd.read_csv(conf['path_to_csv'])
    dataset = MRIDataVolume(
        df, conf['dataset_path'], conf['size'], hist=hist
    )
    dataloader = DataLoader(
        dataset, batch_size=conf['batch_size'], num_workers=conf['workers'], shuffle=False
    )
    return dataloader


def normalize_volume(images):
    """Normalise the intensity values in each modality by scaling by 99 percentile foreground (nonzero) value.

    Parameters
    ----------
    images : tensor
        Tensor containing the volumes

    Returns
    -------
    tensor
        The tensor containing the normalized volumes.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :].reshape(-1)
        i_ = i_[i_ > 0]
        p_99 = torch.quantile(i_, 0.99)
        images[modality, :, :, :] /= p_99

    return images


def hist_norm(images):
    """Applies histogram normalization to the volumes.

    Parameters
    ----------
    images : tensor
        Tensor containing the volumes

    Returns
    -------
    tensor
        The tensor with the histogram normalized.
    """
    for modality in range(images.shape[0]):
        i_ = images[modality, :, :, :]
        mask = np.zeros_like(i_)
        mask[i_ > 0] = 1
        i_ = i_ / np.max(i_)
        i_ = ex.equalize_hist(i_.astype(np.float32), mask=mask, nbins=256)
        i_ *= mask
        images[modality, :, :, :] = i_
    return torch.Tensor(images)


def dice_stitch(pred, target):
    pred_sum = pred.view(-1).sum()
    target_sum = target.view(-1).sum()
    intersection = pred.view(-1).float() @ target.view(-1).float()
    dice = (2 * intersection) / (pred_sum + target_sum)
    return dice


def dice(pred, truth):
    num = 2 * ((pred * truth).sum(dim=(1, 2, 3)).type(torch.float))
    den = (pred.sum(dim=(1, 2, 3)) + truth.sum(dim=(1, 2, 3))).type(torch.float)
    return num / den


def coarse_noise(n, channels, device, noise_size=16, noise_std=0.2, image_size=128):
    noise = torch.normal(
        mean=torch.zeros(n, channels, noise_size, noise_size), std=noise_std
    ).to(device)
    noise = F.interpolate(
        noise,
        size=(image_size, image_size),
        mode="bilinear",
        antialias=False,
        align_corners=True,
    )
    # Roll to randomly translate the generated noise.
    roll_x = random.choice(range(image_size))
    roll_y = random.choice(range(image_size))
    noise = torch.roll(noise, shifts=[roll_x, roll_y], dims=[2, 3])
    return noise


def pyramid_noise_like(n, channels, image_size, discount, device):
    u = transforms.Resize(image_size, antialias=True)
    noise = torch.randn((n, channels, image_size, image_size)).to(device)
    w = image_size
    h = image_size
    for i in range(10):
        r = random.random() * 2 + 2  # Rather than always going 2x,
        w, h = max(1, int(w / (r**i))), max(1, int(h / (r**i)))
        noise += u(torch.randn(n, channels, w, h).to(device)) * discount**i
        if w == 1 or h == 1:
            break  # Lowest resolution is 1x1
    return noise / noise.std()  # Scaled back to roughly unit variance


def random_transform_vectorized(tensor):
    # Random rotations for the entire batch
    angles = torch.randint(0, 24, (tensor.size(0),)) * 15
    sliced_tensor = torch.stack(
        [
            tf.rotate(tensor[i, :, :, :], angles[i].item())
            for i in range(tensor.size(0))
        ]
    )

    # Random horizontal flip for the entire batch
    flip_h = torch.rand(sliced_tensor.size(0)) < 0.5
    for i in range(sliced_tensor.size(0)):
        if flip_h[i]:
            sliced_tensor[i, :, :, :] = tf.hflip(sliced_tensor[i, :, :, :])

    # Random vertical flip for the entire batch
    flip_v = torch.rand(sliced_tensor.size(0)) < 0.5
    for i in range(sliced_tensor.size(0)):
        if flip_v[i]:
            sliced_tensor[i, :, :, :] = tf.vflip(sliced_tensor[i, :, :, :])

    return sliced_tensor

def median_filter_2D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        for j in range(volume.shape[1]):
            volume[i, j, :, :] = medfilt2d(volume[i, j, :, :], kernel_size=kernelsize)
    return torch.Tensor(volume)


def median_filter_3D(volume, kernelsize=5):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Median filtering")
    for i in pbar:
        volume[i] = median_filter(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def connected_components_3d(volume):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Connected components")
    for i in pbar:
        cc_volume = label(volume[i], connectivity=3)
        props = regionprops(cc_volume)
        for prop in props:
            if prop["filled_area"] <= 20:
                volume[i, cc_volume == prop["label"]] = 0
    return torch.Tensor(volume)


def gray_dilation(volume, kernelsize=3):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Gray Dilation")
    for i in pbar:
        volume[i] = grey_dilation(volume[i], size=(kernelsize, kernelsize, kernelsize))
    return torch.Tensor(volume)


def bin_dilation(volume, structure):
    volume = volume.cpu().numpy()
    pbar = tqdm(range(len(volume)), desc="Binray Dilation")
    for i in pbar:
        volume[i] = binary_dilation(volume[i], structure=structure)
    return torch.Tensor(volume)


def norm_tensor(tensor):
    my_max = torch.max(tensor)
    my_min = torch.min(tensor)
    my_tensor = (tensor - my_min) / (my_max - my_min)
    return my_tensor


def gmean(input_x, dim, keepdim=False):
    log_x = torch.log(input_x)
    return torch.exp(torch.mean(log_x, dim=dim, keepdim=keepdim))


def make_dicts(run_name1, run_name2):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name1, run_name2), exist_ok=True)
    os.makedirs(os.path.join("results", run_name1, run_name2), exist_ok=True)
