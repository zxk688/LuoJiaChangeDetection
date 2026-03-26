import os
import numpy as np
import torch
from PIL import Image
import tifffile
from torchvision.transforms import functional as F
from .base import BaseChangeDetectionDataset


num_classes = 3#          蓝色       红色                   
ST_COLORMAP = [[0,0,0],[0,0,255],[255,0,0]]
ST_CLASSES = ['unchanged','Reduced Buildings','Additional buildings' ]

colormap2label = np.zeros(256 ** 3)
for i, cm in enumerate(ST_COLORMAP):
    colormap2label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i


def Color2Index(ColorLabel):
    data = ColorLabel.astype(np.int32)
    idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
    IndexMap = colormap2label[idx]
    #IndexMap = 2*(IndexMap > 1) + 1 * (IndexMap <= 1)
    IndexMap = IndexMap * (IndexMap < num_classes)
    return IndexMap

def Index2Color(pred):
    colormap = np.asarray(ST_COLORMAP, dtype='uint8')
    x = np.asarray(pred, dtype='int32')
    return colormap[x, :]

class S2LookingDataset(BaseChangeDetectionDataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(S2LookingDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.root = self.dataset_cfg['rootdir']
        self.logger_handle = logger_handle
        self.img_A_dir = os.path.join(self.root, mode, 'image1')
        self.img_B_dir = os.path.join(self.root, mode, 'image2')
        self.label_A_dir = os.path.join(self.root, mode, 'label1')
        self.label_B_dir = os.path.join(self.root, mode, 'label2')
        self.fname_list = os.listdir(self.img_A_dir)
        
         
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

    def __getitem__(self, index):
        
        fname = self.fname_list[index]
        
        img_A = self.read(os.path.join(self.img_A_dir,fname))
        img_B = self.read(os.path.join(self.img_B_dir,fname))
        label_A = self.read(os.path.join(self.label_A_dir,fname))
        label_B = self.read(os.path.join(self.label_A_dir,fname))
        label_A = Color2Index(label_A)
        label_B = Color2Index(label_B)
        print(np.unique(label_B))
        img_A = F.to_tensor(img_A)
        img_B = F.to_tensor(img_B)
        label_A = torch.from_numpy(label_A)
        label_B = torch.from_numpy(label_B)

        sample_meta = {'img_A': img_A, 'img_B': img_B, 'label_A': label_A, 'label_B': label_B, 'fname': fname}
        return sample_meta
    
       
    def __len__(self):
        return len(self.fname_list)