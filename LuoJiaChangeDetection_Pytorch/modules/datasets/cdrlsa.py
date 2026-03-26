import glob
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from .base import BaseChangeDetectionDataset
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class CutSwap(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, img2):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)
        img_ = img.clone().detach()
        local = []
        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            img[:,y1: y2, x1: x2] = img2[:,y1: y2, x1: x2]
            img2[:,y1: y2, x1: x2] = img_[:,y1: y2, x1: x2]
            local.append([y1, y2, x1, x2])
        
        return img,img2,local  

class CDRLDataset(BaseChangeDetectionDataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(CDRLDataset, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        
        self.mode = mode
        self.logger_handle = logger_handle
        self.dataset_cfg = dataset_cfg
        root_path=self.dataset_cfg['rootdir']
        dataset = self.dataset_cfg['datasetname']
        self.files = []
            
        if self.mode == 'train':
            self.transforms_A = A.Compose([
                        A.Resize(256,256),
                    #     A.ColorJitter(p=0.5), 
                        A.Normalize(), 
                        ToTensorV2()
                    ])
            self.transforms_B = A.Compose([
                        A.Resize(256, 256),
                        A.Normalize(), 
                        ToTensorV2()
                    ])
            
            self.n_holes = 2
        
        
            for data in dataset.split(','):
                if data!='':
                    self.total_path = os.path.join(root_path, data, mode)
                    self.files += sorted(glob.glob(self.total_path + "/A/*.*")) +\
                                sorted(glob.glob(self.total_path + "/B/*.*"))
        else:
            self.total_path = os.path.join(root_path, dataset, 'test')
            self.transforms = A.Compose([
                        A.Resize(256, 256),
                        A.Normalize(), 
                        ToTensorV2()
                    ])
            self.files = sorted(glob.glob(self.total_path + "/A/*.*"))   
                            
    def __getitem__(self, index):
        
        if self.mode == 'train':
            img_name = self.files[index % len(self.files)].split('/')[-1]
            img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
            img_ori = img_A.copy()
            A2BB2A_path = self.files[index % len(self.files)].split('/'+self.mode+'/')[0]+'_A2B_B2A/'
            if '/A/' in self.files[index % len(self.files)]:
                img_B = cv2.imread(A2BB2A_path+self.mode+ '/A/'+img_name, cv2.IMREAD_COLOR)
            elif '/B/' in self.files[index % len(self.files)]:
                img_B = cv2.imread(A2BB2A_path+self.mode+ '/B/'+img_name, cv2.IMREAD_COLOR)
            
            transformed_A = self.transforms_A(image=img_A)
            transformed_B = self.transforms_B(image=img_B)
            
            img_A = transformed_A["image"]
            img_B = transformed_B["image"]
            
            cutmix_ = CutSwap(n_holes=self.n_holes, length=64)
            img_A_cutmix = img_A.clone().detach()
            img_B_cutmix = img_B.clone().detach()
            img_A_cutmix,img_B_cutmix, local = cutmix_(img_A_cutmix,img_B_cutmix)
            
            return {"A":img_A , "B": img_B, "A_cutmix": img_A_cutmix,"B_cutmix": img_B_cutmix, "local":local}
        elif self.mode == 'test':
            name = self.files[index % len(self.files)].split('/')[-1]
            
            img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
            img_B = cv2.imread(self.files[index % len(self.files)].replace('/A/','/B/'), cv2.IMREAD_COLOR)
            
            transformed_A = self.transforms(image=img_A)
            transformed_B = self.transforms(image=img_B)
            
            img_A = transformed_A["image"]
            img_B = transformed_B["image"]
            
            return {"A": img_A, "B": img_B, 'NAME': name}
    def __len__(self):
        return len(self.files)




# class CDRL_Dataset_train(BaseChangeDetectionDataset):
#     def __init__(self, mode, logger_handle, dataset_cfg):
#         super(CDRL_Dataset_train, self).__init__(mode=mode, logger_handle=logger_handle, dataset_cfg=dataset_cfg)
#         self.transforms_A = transforms_aug
#         self.transforms_B = transforms_ori
#         self.mode = mode
#         self.logger_handle = logger_handle
#         self.dataset_cfg = dataset_cfg
#         root_path=self.dataset_cfg['rootdir']
#         dataset = self.dataset_cfg['datasetname']
#         self.files = []
#         self.n_holes = 2
        
#         for data in dataset.split(','):
#             if data!='':
#                 self.total_path = os.path.join(root_path, data, mode)
#                 self.files += sorted(glob.glob(self.total_path + "/A/*.*")) +\
#                               sorted(glob.glob(self.total_path + "/B/*.*"))
    
 
        
#     def __len__(self):
#         return len(self.files)
    
#     def __getitem__(self, index):
#         img_name = self.files[index % len(self.files)].split('/')[-1]
#         img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
#         img_ori = img_A.copy()
#         A2BB2A_path = self.files[index % len(self.files)].split('/'+self.mode+'/')[0]+'_A2B_B2A/'
#         if '/A/' in self.files[index % len(self.files)]:
#             img_B = cv2.imread(A2BB2A_path+self.mode+ '/A/'+img_name, cv2.IMREAD_COLOR)
#         elif '/B/' in self.files[index % len(self.files)]:
#             img_B = cv2.imread(A2BB2A_path+self.mode+ '/B/'+img_name, cv2.IMREAD_COLOR)
        
#         transformed_A = self.transforms_A(image=img_A)
#         transformed_B = self.transforms_B(image=img_B)
        
#         img_A = transformed_A["image"]
#         img_B = transformed_B["image"]
        
#         cutmix_ = CutSwap(n_holes=self.n_holes, length=64)
#         img_A_cutmix = img_A.clone().detach()
#         img_B_cutmix = img_B.clone().detach()
#         img_A_cutmix,img_B_cutmix, local = cutmix_(img_A_cutmix,img_B_cutmix)
        
#         return {"A":img_A , "B": img_B, "A_cutmix": img_A_cutmix,"B_cutmix": img_B_cutmix, "local":local}

# class CDRL_Dataset_test(BaseChangeDetectionDataset):
#     def __init__(self, mode, logger_handle, dataset_cfg):
#         self.mode = mode
#         self.logger_handle = logger_handle
#         self.dataset_cfg = dataset_cfg
#         root_path=self.dataset_cfg['rootdir']
#         dataset = self.dataset_cfg['datasetname']
#         self.total_path = os.path.join(root_path, dataset, 'test')
#         self.transforms = transforms_ori
#         self.files = sorted(glob.glob(self.total_path + "/A/*.*"))
        
#     def __getitem__(self, index):
#         name = self.files[index % len(self.files)].split('/')[-1]
        
#         img_A = cv2.imread(self.files[index % len(self.files)], cv2.IMREAD_COLOR)
#         img_B = cv2.imread(self.files[index % len(self.files)].replace('/A/','/B/'), cv2.IMREAD_COLOR)
        
#         transformed_A = self.transforms(image=img_A)
#         transformed_B = self.transforms(image=img_B)
        
#         img_A = transformed_A["image"]
#         img_B = transformed_B["image"]
        
#         return {"A": img_A, "B": img_B, 'NAME': name}

#     def __len__(self):
#         return len(self.files)

