import os
import os.path as osp
import numpy as np
import random
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
import torch

class PairedCityscapes(data.Dataset):
    def __init__(self, root_dir, set='train', max_iters=None, img_size=640):
        self.root_dir = root_dir
        self.set = set
        self.img_size = img_size
        # Define paths to CW and SF images and labels
        self.cw_image_dir = osp.join(root_dir, 'CW', 'images', set)
        self.sf_image_dir = osp.join(root_dir, 'SF', 'images', set)
        self.cw_label_dir = osp.join(root_dir, 'CW', 'labels', set)
        self.sf_label_dir = osp.join(root_dir, 'SF', 'labels', set)
        # List image files directly from CW image directory
        self.img_ids = [f for f in os.listdir(self.cw_image_dir) if f.endswith('.png')]

        if max_iters is not None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))

        self.files = []
        for img_id in self.img_ids:
            self.files.append({
                'cw_img': osp.join(self.cw_image_dir, img_id),
                'sf_img': osp.join(self.sf_image_dir, img_id),
                'cw_label': osp.join(self.cw_label_dir, img_id.replace('.png', '.txt')),
                'sf_label': osp.join(self.sf_label_dir, img_id.replace('.png', '.txt')),
                'name': img_id
            })
        self.transform = T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # Load and preprocess images
        cw_img = Image.open(datafiles['cw_img']).convert('RGB')
        sf_img = Image.open(datafiles['sf_img']).convert('RGB')

        # Resize image and convert t tensor
        cw_img = self.transform(cw_img)
        sf_img = self.transform(sf_img)

        # Load and preprocess labels
        cw_label = self.load_yolo_label(datafiles['cw_label'])
        sf_label = self.load_yolo_label(datafiles['sf_label'])

        # Random horizontal flip for augmentation
        if random.random() > 0.5:
            cw_img = T.functional.hflip(cw_img)
            sf_img = T.functional.hflip(sf_img)
            if cw_label['boxes'].numel() > 0:
                cw_label['boxes'][:, [0, 2]] = self.img_size - cw_label['boxes'][:, [2, 0]]
            if sf_label['boxes'].numel() > 0:
                sf_label['boxes'][:, [0, 2]] = self.img_size - sf_label['boxes'][:, [2, 0]]
        return cw_img, sf_img, cw_label, sf_label, datafiles['name'], 'CW', 'SF'

    def load_yolo_label(self, label_path):
        boxes, labels = [], []
        if osp.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x, y, w, h = map(float, line.strip().split())
                    # Convert normalized coordinates to absolute (640x640)
                    x_min = (x - w/2) * self.img_size
                    y_min = (y - h/2) * self.img_size
                    x_max = (x + w/2) * self.img_size
                    y_max = (y + h/2) * self.img_size
                    boxes.append([x_min, y_min, x_max, y_max])
                    labels.append(int(class_id))
        return {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

    def collate_fn(self, batch):
        cw_imgs, sf_imgs, cw_labels, sf_labels, names, cw_domains, sf_domains = zip(*batch)

        # Stack image tensors
        cw_imgs = torch.stack(cw_imgs, dim=0)
        sf_imgs = torch.stack(sf_imgs, dim=0)

        # Keep labels as lists of dicts
        return cw_imgs, sf_imgs, list(cw_labels), list(sf_labels), list(names), list(cw_domains), list(sf_domains)