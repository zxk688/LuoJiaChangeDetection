import numpy as np
import os
from PIL import Image
import tifffile
from luojianet import Tensor
import luojianet.ops as F

# data root
# ├─A
# ├─B
# ├─label
# └─list
   # ├─train.txt
   # ├─test.txt
   # └─eval.txt

def load_img_name_list(list_path):
        img_name_list = np.loadtxt(list_path, dtype=str)
        img_name_list = img_name_list.tolist()
        return img_name_list

class X2ViewDataset():
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(X2ViewDataset, self).__init__()
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        img_folder_names = self.dataset_cfg['img_folder_names']
        list_folder_name = self.dataset_cfg['list_folder_name']
        self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], img_folder_names[0])
        self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], img_folder_names[1])
        self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], img_folder_names[2])
        self.list_path = os.path.join(self.dataset_cfg['rootdir'], list_folder_name, f'{mode}_label.txt')
                                          
        self.img_name_list= load_img_name_list(self.list_path)

        self.class_names = ['Background', 'Intact', 'Minor', 'Major', 'Destroyed']
        
    def __getitem__(self, index):
        fname = self.img_name_list[index]
        fname = os.path.split(fname)[-1]
        pre_img_path=os.path.join(self.pre_change_path,fname.replace('post_disaster_target','pre_disaster'))
        post_img_path=os.path.join(self.post_change_path,fname.replace('post_disaster_target','post_disaster'))
        change_gt_path=os.path.join(self.change_gt_path,fname)

        pre_img = self.read(pre_img_path)
        post_img = self.read(post_img_path)
        change_gt = self.read(change_gt_path)

        pre_img = (pre_img.transpose((2, 0, 1)).astype(np.float32))/255
        post_img = (post_img.transpose((2, 0, 1)).astype(np.float32))/255
        # change_gt = change_gt.astype(np.float32)
        
        change_gt = np.eye(5)[change_gt]
        change_gt = (change_gt.transpose((2, 0, 1)).astype(np.float32))

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
        return len(self.img_name_list)

# import numpy as np
# import os
# from PIL import Image
# import tifffile
# # data root
# # ├─A
# # ├─B
# # ├─label
# # └─list
#    # ├─train.txt
#    # ├─test.txt
#    # └─eval.txt

# def load_img_name_list(list_path):
#         img_name_list = np.loadtxt(list_path, dtype=str)
#         img_name_list = img_name_list.tolist()
#         return img_name_list

# class X2ViewDataset():
#     def __init__(self, mode, logger_handle, dataset_cfg):
#         super(X2ViewDataset, self).__init__()
#         self.mode = mode
#         self.dataset_cfg = dataset_cfg
#         self.logger_handle = logger_handle
#         img_folder_names = self.dataset_cfg['img_folder_names']
#         list_folder_name = self.dataset_cfg['list_folder_name']
#         self.pre_change_path = os.path.join(self.dataset_cfg['rootdir'], img_folder_names[0])
#         self.post_change_path = os.path.join(self.dataset_cfg['rootdir'], img_folder_names[1])
#         self.change_gt_path = os.path.join(self.dataset_cfg['rootdir'], img_folder_names[2])
#         self.list_path = os.path.join(self.dataset_cfg['rootdir'], list_folder_name, f'{mode}_label.txt')
                                          
#         self.img_name_list= load_img_name_list(self.list_path)

#         self.class_names = ['Background', 'Intact', 'Minor', 'Major', 'Destroyed']
        
#     def __getitem__(self, index):
#         fname = self.img_name_list[index]
#         fname = os.path.split(fname)[-1]
#         pre_img_path=os.path.join(self.pre_change_path,fname.replace('post_disaster_target','pre_disaster'))
#         post_img_path=os.path.join(self.post_change_path,fname.replace('post_disaster_target','post_disaster'))
#         change_gt_path=os.path.join(self.change_gt_path,fname)

#         pre_img = self.read(pre_img_path)
#         post_img = self.read(post_img_path)
#         change_gt = self.read(change_gt_path)

#         pre_img = (pre_img.transpose((2, 0, 1)).astype(np.float32))/255
#         post_img = (post_img.transpose((2, 0, 1)).astype(np.float32))/255
#         change_gt = np.expand_dims(change_gt.astype(np.float32), axis=0)
#         if self.mode == 'train':
#             return [pre_img, post_img], change_gt
#         else:
#             return [pre_img, post_img], change_gt, fname
    
#     def read(self, image_path):
#         assert (image_path is not None) and os.path.exists(image_path)
        
#         try:
#             if image_path.endswith('.png') or image_path.endswith('.jpg'):
#                 sample_meta = np.array(Image.open(image_path))
#             elif image_path.endswith('.tif'):
#                 sample_meta = tifffile.imread(image_path)
#             else:
#                 raise ValueError(f"Unsupported file type: {image_path}")
#         except Exception as e:
#             print(f"Error reading image: {e}")
#             return None
#         return sample_meta
    
    
#     def __len__(self):
#         return len(self.img_name_list)

