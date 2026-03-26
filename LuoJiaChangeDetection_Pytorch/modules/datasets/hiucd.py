import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import tifffile
from .base import BaseChangeDetectionDataset

# data root
# ├─train
   # ├─A
   # ├─B
   # └─label
# ├─test
   # ├─A
   # ├─B
   # └─label
# └─eval
   # ├─A
   # ├─B
   # └─label

    
class HiUCDDataset(BaseChangeDetectionDataset):
    num_classes = 2
    classnames = ['change', 'unchanged']
    # palette = [(0, 0, 0), (255, 255, 255)]
    palette = BaseChangeDetectionDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(HiUCDDataset, self).__init__(mode=mode , logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.img_folder_names = img_folder_names
        self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, 'image/2017')
        self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, 'image/2017')
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, 'mask_merge/2017_2018')
        
        self.pre_filenames = []
        for root, _, filenames in os.walk(self.change_gt_path):
            for filename in filenames:
                self.pre_filenames.append(os.path.join(root, filename))
        
        self.post_filenames = []
        for root, _, filenames in os.walk(self.change_gt_path):
            for filename in filenames:
                self.post_filenames.append(os.path.join(root, filename))
                
        self.gt_filenames = []
        for root, _, filenames in os.walk(self.change_gt_path):
            for filename in filenames:
                self.gt_filenames.append(os.path.join(root, filename))
                                
    def __getitem__(self, index):
        
        pre_img_path=self.pre_filenames[index]
        post_img_path=self.post_filenames[index]
        gt_path=self.gt_filenames[index]
        pre_img = self.read(pre_img_path)
        post_img = self.read(post_img_path)
        pre_gt = self.read(gt_path)[:,:,0]
        post_gt = self.read(gt_path)[:,:,1]
        change_gt = self.read(gt_path)[:,:,2]
        
        # transform = transforms.Compose([transforms.ToTensor()])
        # pre_tensor = transform(pre_img)
        # post_tensor = transform(post_img)
        # gt_tensor = transform(change_gt)



        sample_meta = {'image1': pre_img, 'image2': post_img, 
                       'pre_gt': pre_gt, 'post_gt': post_gt, 
                       'gt': change_gt, 'fname': ''}
        sample_meta = self.synctransforms(sample_meta)

        return sample_meta
        
    
    def read(self, image_path):
        assert (image_path is not None) and os.path.exists(image_path)
        
        try:
            if image_path.endswith('.png') or image_path.endswith('.jpg'):
                sample_meta = np.array(Image.open(image_path))
            elif image_path.endswith('.tif'):
                sample_meta = tifffile.imread(image_path)
            else:
                raise ValueError(f"Unsupported file type: {image_path}")
        except Exception as e:
            print(f"Error reading image: {e}")
            return None

        return sample_meta

    def __len__(self):
        return len(self.gt_filenames)
    




