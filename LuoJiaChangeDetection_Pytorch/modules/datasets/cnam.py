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

class SemanticCDDataset(BaseChangeDetectionDataset):
    num_classes = 5
    classnames = ['Background', 'Impervious surface', 
                  'Bare ground', 'Vegetation', 'Water bodies', 'Other']
    # palette = [(0, 0, 0), (255, 255, 255)]
    palette = BaseChangeDetectionDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(SemanticCDDataset, self).__init__(mode=mode , logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[0])
        self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[1])
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[2])
        self.fname_list=os.listdir(self.pre_change_path)

    def __getitem__(self, index):

        fname = self.fname_list[index]

        pre_img_path=os.path.join(self.pre_change_path,fname)
        post_img_path=os.path.join(self.post_change_path,fname)
        change_gt_path=os.path.join(self.change_gt_path,fname)
        pre_img = self.read(pre_img_path)
        post_img = self.read(post_img_path)
        change_gt = self.read(change_gt_path)
        
        # 
        transform = transforms.Compose([transforms.ToTensor()])
        pre_tensor = transform(pre_img)
        post_tensor = transform(post_img)
        gt_tensor=torch.from_numpy(change_gt)

        sample_meta = {'image1': pre_tensor, 'image2': post_tensor, 'gt': gt_tensor, 'fname': fname}
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
    




