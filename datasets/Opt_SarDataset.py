from torch.utils.data import Dataset
import os
from torchvision import transforms
import numpy as np
import torch
from utils.Common import common
from datasets.Augmentor import AugToolkit


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(img.astype(np.int32)).long()

normalize_opt = transforms.Normalize([0.4171278135567563,
                                      0.38524851176305586,
                                      0.31584864673690705,
                                      0.3920406749409602],
                                     [0.05900967942210461,
                                      0.06627405308424451,
                                      0.07930850013605899,
                                      0.1211643350940716])

normalize_sar = transforms.Normalize([0.21132847661424917], [0.18894593775525773])

img_opt_transform = transforms.Compose([transforms.ToTensor(), normalize_opt])

img_sar_transform = transforms.Compose([transforms.ToTensor(), normalize_sar])

mask_transform = MaskToTensor()


class Opt_SarDataset(Dataset):
    def __init__(self, class_name, root, mode=None,
                 img_opt_transform=img_opt_transform,
                 img_aux_transform=img_sar_transform,
                 mask_transform=mask_transform,
                 aug_params=None,
                 ):
        # 数据相关
        self.mode = mode
        self.class_names = class_name
        self.img_opt_transform = img_opt_transform
        self.img_sar_transform = img_aux_transform
        self.mask_transform = mask_transform
        self.sync_img_mask = []
        self.output_names = []

        img_opt_dir = os.path.join(root, 'opt')
        img_sar_dir = os.path.join(root, 'sar')
        mask_dir = os.path.join(root, 'lbl')

        for img_filename in os.listdir(mask_dir):
            img_mask_pair = (os.path.join(img_opt_dir, img_filename.replace('.png', '.tif')),
                             os.path.join(img_sar_dir, img_filename.replace('.png', '.tif')),
                             os.path.join(mask_dir, img_filename))
            self.sync_img_mask.append(img_mask_pair)
            self.output_names.append(img_filename)

        self.aug_toolkit = None
        if aug_params is not None:
            self.aug_toolkit = AugToolkit(aug_params)

        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        img_opt_path, img_sar_path, mask_path = self.sync_img_mask[index]
        output_names = self.output_names[index]

        img_opt = common.gdal_to_numpy(img_opt_path)
        img_sar = common.gdal_to_numpy(img_sar_path)
        mask = common.gdal_to_numpy(mask_path)

        img = np.append(img_opt, img_sar, axis=2)
        img = np.append(img, mask, axis=2)

        if self.mode == 'train':
            if self.aug_toolkit is not None:
                img = self.aug_toolkit.run(img)

        mask = img[:, :, -1].copy()
        img_opt = img[:, :, :4].copy()
        img_sar = img[:, :, 4:-1].copy()

        if self.img_sar_transform is not None:
            img_opt = self.img_opt_transform(img_opt)
            img_sar = self.img_sar_transform(img_sar)
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return img_opt, img_sar, mask, output_names

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names


if __name__ == "__main__":
    mean_sar = common.load_mean_file(
        r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\sar_mean_std\mean_value.txt')
    std_sar = common.load_mean_file(
        r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\sar_mean_std\std_value.txt')
    mean_opt = common.load_mean_file(
        r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\opt_mean_std\mean_value.txt')
    std_opt = common.load_mean_file(
        r'G:\deepl_datasets\语义分割公开数据集\run\whu_Sar_Opt\info\opt_mean_std\std_value.txt')

    print(mean_sar)
    print(std_sar)
    print(mean_opt)
    print(std_opt)
