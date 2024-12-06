import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image
import h5py
from imgaug import augmenters as iaa
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, message=".*np.bool*")


class Whole_Slide_Bag(Dataset):
    def __init__(self, file_path, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            roi_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.roi_transforms = img_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        with h5py.File(self.file_path, "r") as hdf5_file:
            dset = hdf5_file['imgs']
            for name, value in dset.attrs.items():
                print(name, value)

        print('transformations:', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}


class Whole_Slide_Bag_FP(Dataset):
    def __init__(self, file_path, wsi, img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            img_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)

        self.summary()

    def augmentor(self, image):
        flip_aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
            ])
        ], random_order=True)

        crop_aug = iaa.Sometimes(0.5, iaa.Sequential([
            iaa.OneOf([
                iaa.CropToFixedSize(144, 144),
                iaa.CropToFixedSize(160, 160),
                iaa.CropToFixedSize(176, 176),
                iaa.CropToFixedSize(192, 192),
                iaa.CropToFixedSize(208, 208),
                iaa.CropToFixedSize(224, 224),
            ])
        ]))

        pad_aug = iaa.PadToFixedSize(width=224, height=224)

        augment_image = iaa.Sequential([flip_aug, crop_aug, pad_aug])

        image_aug = augment_image.augment_image(image)
        return image_aug

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]
        img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')

        img = np.array(img)
        img = self.augmentor(img)
        img = self.roi_transforms(Image.fromarray(img))

        return {'img': img, 'coord': coord}


class Dataset_All_Bags(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]