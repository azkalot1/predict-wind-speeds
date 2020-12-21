import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread


class WindDataset(Dataset):
    def __init__(
        self,
        images,
        transform,
            wind_speed=None):
        self.images = images
        self.wind_speed = wind_speed
        self.transform = transform

    def __len__(self):
        return(len(self.images))

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        image = imread(self.images[idx])
        image = np.expand_dims(image, 2) / 255
        augmented = self.transform(image=image)
        image = augmented['image']
        image = image.transpose(2, 0, 1)  # channels first
        data = {'features': torch.from_numpy(image.copy()).float()}
        if self.wind_speed is not None:
            wind_speed = np.zeros(1)
            wind_speed[0] = self.wind_speed[idx]
            data['target'] = torch.from_numpy(wind_speed).float()
        return(data)
