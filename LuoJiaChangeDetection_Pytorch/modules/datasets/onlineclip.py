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
# └─val
   # ├─A
   # ├─B
   # └─label

class OnlineClipDataset(BaseChangeDetectionDataset):
    num_classes = 2
    classnames = ['change', 'unchanged']
    # palette = [(0, 0, 0), (255, 255, 255)]
    palette = BaseChangeDetectionDataset.randompalette(num_classes)
    assert num_classes == len(classnames) and num_classes == len(palette)
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(OnlineClipDataset, self).__init__(mode=mode , logger_handle=logger_handle, dataset_cfg=dataset_cfg)
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        self.img_folder_names = img_folder_names
        self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[0])
        self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[1])
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], mode, img_folder_names[2])
        
        self.fname_list = self.pathcounting(overlap = 0, split_size = 128,
            inputpath = self.pre_change_path)
    

    def __getitem__(self, index):

        fname = self.fname_list[index]
        
        pre_img_path=os.path.join(self.pre_change_path,fname)
        post_img_path=os.path.join(self.post_change_path,fname)
        change_gt_path=os.path.join(self.change_gt_path,fname)
        
        pre_img_path_org = pre_img_path[:pre_img_path.rfind('_')]+os.path.splitext(pre_img_path)[1]
        ind_pre = os.path.splitext(pre_img_path)[0][pre_img_path.rfind('_')+1:]
        post_img_path_org = post_img_path[:post_img_path.rfind('_')]+os.path.splitext(post_img_path)[1]
        ind_post = os.path.splitext(post_img_path)[0][post_img_path.rfind('_')+1:]
        change_gt_path_org = change_gt_path[:change_gt_path.rfind('_')]+os.path.splitext(change_gt_path)[1]
        ind_gt = os.path.splitext(change_gt_path)[0][change_gt_path.rfind('_')+1:] 
        
        pre_img = self.findthepatch(original_name = pre_img_path_org,index = int(ind_pre))
        post_img = self.findthepatch(original_name = post_img_path_org,index = int(ind_post))
        change_gt = self.findthepatch(original_name = change_gt_path_org,index = int(ind_gt))
        
        transform = transforms.Compose([transforms.ToTensor()])
        pre_tensor = transform(pre_img)
        post_tensor = transform(post_img)
        gt_tensor = transform(change_gt)

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
    
    def start_points(self, size, split_size, overlap=0):
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points


    def findthepatch(self, overlap = 0,
        split_size = 128,
        original_name='', index=0):
        img = np.asarray(self.read(original_name))

        X_points = self.start_points(img.shape[0], split_size, overlap)
        Y_points = self.start_points(img.shape[1], split_size, overlap)
        
        ind = 0
        for i in Y_points:
            for j in X_points:  
                if len(np.shape(img))<3:
                    split = img[i:i + split_size, j:j + split_size]
                else:    
                    split = img[i:i + split_size, j:j + split_size,:]
                if(index==ind):
                    return split   
                ind += 1


    def pathcounting(self, overlap = 0,
        split_size = 128,
        inputpath = None):
        patch_list = []

        img_list = os.listdir(inputpath)
        for img_name in img_list:
            
            img = os.path.join(inputpath,img_name)
            img = self.read(img)
            img_h, img_w, _ = img.shape
            X_points = self.start_points(img_w, split_size, overlap)
            Y_points = self.start_points(img_h, split_size, overlap)
            id = 0
            for i in Y_points:
                for j in X_points:  
                    patch_name = os.path.splitext(img_name)[0]+ '_'+str(id)+os.path.splitext(img_name)[1] 
                    patch_list.append(patch_name)
                    id += 1
        return  patch_list

