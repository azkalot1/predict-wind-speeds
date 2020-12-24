import torch
from torch.utils.data import Dataset
import numpy as np
from skimage.io import imread


class WindDataset(Dataset):
    def __init__(
        self,
        images,
        folder,
        transform,
        load_n=1,
        type_agg='simple',
            wind_speed=None):
        self.images = images
        self.folder = folder
        self.wind_speed = wind_speed
        self.transform = transform
        self.type_agg = type_agg
        self.load_n = load_n

    def __len__(self):
        return(len(self.images))

    def get_path_load(self, idx):
        image_id = self.images[idx]
        image_order = int(image_id.split('_')[-1].split('.')[0])
        image_storm = image_id.split('_')[0]
        last_prev_idx = max(0, image_order-self.load_n+1)
        prev_load = [*range(last_prev_idx, image_order+1)]
        prev_load = [last_prev_idx] * (self.load_n-len(prev_load)) + prev_load
        prev_load = [self.folder + '/' + image_storm+'_'+str(x).zfill(3)+'.jpg' for x in prev_load]
        assert(len(prev_load) == self.load_n)
        return prev_load

    def __getitem__(self, idx):
        """Will load the mask, get random coordinates around/with the mask,
        load the image by coordinates
        """
        # get image id and image order
        images_to_load = self.get_path_load(idx)
        image = [imread(image_path) for image_path in images_to_load]
        image = np.stack(image, axis=0) / 255
        if self.type_agg == 'minmaxmean':
            im_max = image.max(axis=0)
            im_mean = image.mean(axis=0)
            im_min = image.min(axis=0)
            image = np.stack([im_max, im_mean, im_min], axis=0)
        image = image.transpose((1, 2, 0))
        augmented = self.transform(image=image)
        image = augmented['image']
        image = image.transpose(2, 0, 1)  # channels first
        data = {'features': torch.from_numpy(image.copy()).float()}
        if self.wind_speed is not None:
            wind_speed = np.zeros(1)
            wind_speed[0] = self.wind_speed[idx]
            data['target'] = torch.from_numpy(wind_speed).float()
        return(data)
