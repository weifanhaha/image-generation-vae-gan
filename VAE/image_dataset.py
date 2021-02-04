#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T
import glob

class ImageDataset(Dataset):
    def __init__(self, mode, predict_img_path=''):
        assert mode in ["train", "val", "test"]

        self.mode = mode
        self.init_img_paths(mode)
        
        self.len = len(self.img_paths)
        
    def init_img_paths(self, mode):
        train_img_path = "../hw3_data/face/train/"
        test_img_path = "../hw3_data/face/test/"

        train_img_paths = glob.glob(train_img_path + '*.png')
        test_img_paths = glob.glob(test_img_path + '*.png')
        
        if mode == 'train':
            self.img_paths = train_img_paths[:36000]
        elif mode == "val":
            self.img_paths = train_img_paths[36000:]
        elif mode == 'test':
            self.img_paths = sorted(test_img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_name = img_path.split('/')[-1]

        # get image tensor
        image = Image.open(img_path).convert("RGB")
        image_tensor = T.ToTensor()(image)
        
        return image_tensor

    def __len__(self):
        return self.len

