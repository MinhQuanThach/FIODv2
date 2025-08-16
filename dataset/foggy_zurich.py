import os
import os.path as osp
import numpy as np
import random
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import torch

class FoggyZurich(data.Dataset):
    def __init__(self, root_dir, set='train', max_iters=None, img_size=640):
        self.root_dir = root_dir
        self.set = set
        self.img_size = img_size
        self.rf_image_dir = osp.join(root_dir, 'RF', 'images', set)
        self.img_ids = [f for f in os.listdir(self.rf_image_dir) if f.endswith('.png')]
        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []
        for img_id in self.img_ids:
            self.files.append({
                'rf_img': osp.join(self.rf_image_dir, img_id),
                'name': img_id
            })
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        rf_img = Image.open(datafiles['rf_img']).convert('RGB')
        rf_img = self.transform(rf_img)
        if random.random() > 0.5:
            rf_img = T.functional.hflip(rf_img)
        return rf_img, datafiles['name'], 'RF'

    def collate_fn(self, batch):
        """
        Hàm này được sử dụng để gom các mẫu trong batch lại với nhau.
        """
        fog_images, img_names, domains = zip(*batch)
        fog_images = torch.stack(fog_images, 0)
        return fog_images, img_names, list(domains)