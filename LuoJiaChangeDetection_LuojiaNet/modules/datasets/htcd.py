import numpy as np
import os
import cv2
import tifffile
from PIL import Image

 
class HTCDDataset():
    # img1-sat img2-uav
    def __init__(self, mode, logger_handle, dataset_cfg):
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        self.dir = self.dataset_cfg['rootdir']
        self.images_list = os.listdir(os.path.join(self.dir, mode, 'image1'))
        self.sat_mean = np.array([66, 71, 74], np.uint8)
        self.uav_mean = np.array([73, 81, 79], np.uint8)

    def __getitem__(self, idx):
        # img1-sat img2-uav
        filename = self.images_list[idx]
        img_sat_file = os.path.join(self.dir, self.mode, 'image2', filename)
        edge_sat_file = os.path.join(self.dir, self.mode, 'edges_uav', filename)
        img_uav_file = os.path.join(self.dir, self.mode, 'image1', filename)
        edge_uav_file = os.path.join(self.dir, self.mode, 'edges_sat', filename)
        label_file = os.path.join(self.dir, self.mode, 'label', filename)

        img_sat = self.read(img_sat_file).astype(np.int16)
        img_sat -= self.sat_mean
        img_size = img_sat.shape[:2]
        edge_sat = self.read(edge_sat_file)
        edge_sat = cv2.resize(edge_sat, img_size).astype(np.int16)
        img_sat = np.concatenate((img_sat, edge_sat[..., np.newaxis]), axis=2)
        img_sat = (img_sat.transpose((2, 0, 1)).astype(np.float32))

        img_uav = self.read(img_uav_file)
        img_uav -= self.uav_mean
        edge_uav = self.read(edge_uav_file)
        edge_uav = cv2.resize(edge_uav, (256,256)).astype(np.int16)
        img_uav = np.concatenate((img_uav, edge_uav[..., np.newaxis]), axis=2)
        img_uav = (img_uav.transpose((2, 0, 1)).astype(np.float32))

        gt = self.read(label_file)
        gt = gt[:,:,0]
        gt[gt > 0] = 1
        gt = np.expand_dims(gt,0)

        labels = [img_sat, gt]

        return img_uav, labels 
    
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
        return len(self.images_list)

# train_data = dataset = HTCD('/media/lscsc/nas/xianping/ZXK/CDDemo/CDFramework/data/HTCD/',mode='train')
# train_dataloader = DataLoader(train_data, batch_size=2,
#                                   shuffle=True, num_workers=4, pin_memory=True)
# for i, data in enumerate(train_dataloader):
#     x1, x2, lbl = data
#     print(x1.shape,x2.shape,lbl.shape)