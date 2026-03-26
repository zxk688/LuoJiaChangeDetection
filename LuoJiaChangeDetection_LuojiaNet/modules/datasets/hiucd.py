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

    
class HiUCDDataset():  
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(HiUCDDataset, self).__init__()
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.img_folder_names = img_folder_names
        self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[0])
        self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[1])
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[2])
        
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
        
        pre_img = (pre_img.transpose((2, 0, 1)).astype(np.float32))/255
        post_img = (post_img.transpose((2, 0, 1)).astype(np.float32))/255
        pre_gt = np.expand_dims(pre_gt.astype(np.float32), axis=0)
        post_gt = np.expand_dims(post_gt.astype(np.float32), axis=0)
        change_gt = np.expand_dims(change_gt.astype(np.float32), axis=0)

        images = [pre_img,post_img]
        labels = [pre_gt,post_gt,change_gt]

        return images,labels
        
    
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
    




