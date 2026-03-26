import torch
import numpy as np
import os
import cv2
import tifffile
from PIL import Image
from .base import BaseChangeDetectionDataset

# data root
# ├─A
# ├─B
# ├─label
# └─list
   # ├─train.txt
   # ├─test.txt
   # └─eval.txt

class OpenWUSU512(BaseChangeDetectionDataset):
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(OpenWUSU512, self).__init__(mode=mode , logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        self.imgs_list = []
        self.gt_list = []

        self.cd_type = self.dataset_cfg['cd_type']
        city_list = os.listdir(os.path.join(self.dataset_cfg['rootdir'], mode))
        
        for city in city_list:
            gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, city, 'change', self.cd_type)
            img_path = os.path.join(self.dataset_cfg['rootdir'], mode, city, 'imgs')
            for gt_file in os.listdir(gt_path):
                self.gt_list.append(os.path.join(gt_path, gt_file))
            for img_file in os.listdir(img_path): 
                self.imgs_list.append(os.path.join(img_path, img_file))
                
            

        
    def __getitem__(self, index):

        gt_file_path = self.gt_list[index]
        dir_name, full_file_name = os.path.split(gt_file_path) 
        pre_name = full_file_name[:full_file_name.find('_')-2] + full_file_name[full_file_name.find('_'):] 
        post_name = full_file_name[:full_file_name.find('_')-4] + full_file_name[full_file_name.find('_')-2:]

        image1 = self.read(os.path.join(dir_name.replace('change','imgs').strip(self.cd_type), pre_name))
        image2 = self.read(os.path.join(dir_name.replace('change','imgs').strip(self.cd_type), post_name))
        cd_target = self.read(gt_file_path)

        sample_meta = {'image1': image1, 'image2': image2, 'gt': cd_target,'fname':full_file_name}
        sample_meta = self.synctransforms(sample_meta)

        return sample_meta
        
    def read(self, image_path):
        assert (image_path is not None) and os.path.exists(image_path)
        
        try:
            if image_path.endswith('.png') or image_path.endswith('.jpg'):
                sample_meta = np.array(Image.open(image_path))
            elif image_path.endswith('.tif'):
                sample_meta = np.array(Image.open(image_path))
            else:
                raise ValueError(f"Unsupported file type: {image_path}")
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

        return sample_meta

    def __len__(self):
        return len(self.gt_list)







