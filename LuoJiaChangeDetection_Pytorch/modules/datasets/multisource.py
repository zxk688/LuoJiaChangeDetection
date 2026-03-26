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

class MultisourceDataset(BaseChangeDetectionDataset):
    num_classes = 2
    classnames = ['change', 'unchanged']
    # palette = [(0, 0, 0), (255, 255, 255)]
    palette = BaseChangeDetectionDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(MultisourceDataset, self).__init__(mode=mode , logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.img_folder_names = img_folder_names
       
        self.rgb1_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[0])
        self.rgb2_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[1])
        self.ms_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[2])
        self.sar_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[3])
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[4])
        self.fname_list = os.listdir(self.change_gt_path)

    def __getitem__(self, index):

        fname = self.fname_list[index]

        rgb1_img = self.read(os.path.join(self.rgb1_path,fname))
        rgb2_img = self.read(os.path.join(self.rgb2_path,fname))
        ms_img = self.read(os.path.join(self.ms_path,fname)).astype(np.float16)
        ms_img1 = ms_img[:, :, :7]
        ms_img2 = ms_img[:, :, 7:]
        sar_img = self.read(os.path.join(self.sar_path,fname)).astype(np.float16)
        sar_img1 = sar_img[:, :, :2]
        sar_img2 = sar_img[:, :, 2:]
        gt = self.read(os.path.join(self.change_gt_path,fname))
        
        transform = transforms.Compose([transforms.ToTensor()])
        rgb1_img = transform(rgb1_img)
        rgb2_img = transform(rgb2_img)
        ms_img1 = transform(ms_img1)
        ms_img2 = transform(ms_img2)
        sar_img1 = transform(sar_img1)
        sar_img2 = transform(sar_img2)
        gt = transform(gt)

        sample_meta = {'rgb1': rgb1_img, 'rgb2': rgb2_img, 
                       'ms1': ms_img1, 'ms2': ms_img2, 
                       'sar1': sar_img1, 'sar2': sar_img2, 
                       'gt': gt, 'fname': fname}
        # sample_meta = self.synctransforms(sample_meta)

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
        return len(self.fname_list)
    




