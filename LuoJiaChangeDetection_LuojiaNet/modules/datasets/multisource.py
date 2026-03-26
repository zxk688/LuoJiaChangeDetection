import os
import numpy as np
from PIL import Image
import tifffile


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

class MultisourceDataset():   
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
        
        rgb1_img = (rgb1_img.transpose((2, 0, 1)).astype(np.float32))/255
        rgb2_img = (rgb2_img.transpose((2, 0, 1)).astype(np.float32))/255
        ms_img1 = (ms_img1.transpose((2, 0, 1)).astype(np.float32))/255
        ms_img2 = (ms_img2.transpose((2, 0, 1)).astype(np.float32))/255
        sar_img1 = np.expand_dims(sar_img1.astype(np.float32), axis=0)
        sar_img2 = np.expand_dims(sar_img2.astype(np.float32), axis=0)
        gt = np.expand_dims(gt.astype(np.float32), axis=0)/255
        
        images = [rgb1_img,rgb2_img,ms_img1,ms_img2,sar_img1,sar_img2]

        return images,gt
        
    
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
    




