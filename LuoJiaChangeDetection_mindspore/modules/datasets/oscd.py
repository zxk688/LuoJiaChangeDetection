import os
import numpy as np
from PIL import Image
import tifffile
import glob
import cv2
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

class OSCDDataset():  
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(OSCDDataset, self).__init__()
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.img_folder_names = img_folder_names

        # self.ms_path1 = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[0])
        # self.ms_path2 = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[1])
        # self.sar_path1 = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[2])
        # self.sar_path2 = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[3])
        # self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[4])
        self.fname_list = os.listdir(os.path.join(self.dataset_cfg['rootdir'], mode))
    
    def __getitem__(self, index):

        fname = self.fname_list[index]
        
        ms_img1, ms_img2 = list(), list()
        ms_path1 = os.path.join(self.dataset_cfg['rootdir'], self.mode, fname, self.img_folder_names[0])
        ms_list1 = glob.glob(ms_path1+'/*.tif')
        for i in ms_list1:
            ms_img1.append(cv2.resize(self.read(i),(512,512)))
        ms_img1 = np.array(ms_img1)
        ms_path2 = os.path.join(self.dataset_cfg['rootdir'], self.mode, fname, self.img_folder_names[1])
        ms_list2 = glob.glob(ms_path2+'/*.tif')
        for i in ms_list2:
            ms_img2.append(cv2.resize(self.read(i),(512,512)))
        ms_img2 = np.array(ms_img2)  
        
         
        sar_path1 = os.path.join(self.dataset_cfg['rootdir'], self.mode, fname, self.img_folder_names[2])
        sar_name1 = glob.glob(sar_path1+'/*.tif')[0]
        sar_img1 = cv2.resize(self.read(sar_name1),(512,512))
        sar_img1 = sar_img1.transpose((2, 0, 1))
        sar_path2 = os.path.join(self.dataset_cfg['rootdir'], self.mode, fname, self.img_folder_names[3])
        sar_name2 = glob.glob(sar_path2+'/*.tif')[0]
        sar_img2 = cv2.resize(self.read(sar_name2),(512,512))
        sar_img2 = sar_img2.transpose((2, 0, 1))
        
        gt_path = os.path.join(self.dataset_cfg['rootdir'], self.mode, fname, self.img_folder_names[4])
        gt_name = glob.glob(gt_path+'/*.png')[0]
        gt = self.read(gt_name)
        gt = cv2.resize(gt,(512,512))
        gt = gt.transpose((2, 0, 1))[0,:,:]/255

        images = np.concatenate((sar_img1,sar_img2,ms_img1,ms_img2),0)/255

        return images, gt
        
    
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
    




