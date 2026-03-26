import os
import numpy as np
from PIL import Image
import tifffile
import glob
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

class FolderSplittedDataset():
    # num_classes = 2
    # classnames = ['change', 'unchanged']
    # # palette = [(0, 0, 0), (255, 255, 255)]
    # palette = BaseChangeDetectionDataset.randompalette(num_classes)
    # assert num_classes == len(classnames) and num_classes == len(palette)
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(FolderSplittedDataset, self).__init__()
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.img_folder_names = img_folder_names
        self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[0])
        self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[1])
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[2])
        self.fname_list=[f for f in os.listdir(self.pre_change_path) if not f.startswith('.')]
        
    def __getitem__(self, index):

        fname = self.fname_list[index]
        pre_img_path=os.path.join(self.pre_change_path,fname)
        post_img_path=os.path.join(self.post_change_path,fname)
        change_gt_path=os.path.join(self.change_gt_path,fname)
        pre_img = self.read(pre_img_path)
        post_img = self.read(post_img_path)
        change_gt = self.read(change_gt_path)
        
        pre_img = ((pre_img/255.0).transpose((2, 0, 1)).astype(np.float32))
        post_img = ((post_img/255.0).transpose((2, 0, 1)).astype(np.float32))
        if np.max(change_gt) == 255:
            change_gt = np.expand_dims(change_gt.astype(np.float32)/255, axis=0)
        else: 
            change_gt = np.expand_dims(change_gt.astype(np.float32), axis=0)
        images = np.concatenate((pre_img, post_img), axis=0)  
        if self.mode == 'train':
            return [pre_img, post_img], change_gt
        else: 
            return [pre_img, post_img], change_gt, fname
        
    
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


    




