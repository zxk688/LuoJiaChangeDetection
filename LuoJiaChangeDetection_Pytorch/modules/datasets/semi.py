import os
import numpy as np
from PIL import Image
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        self.dataset_cfg = dataset_cfg[mode]
        self.root = self.dataset_cfg['data_dir']
        self.split = mode
        self.percnt_lbl = self.dataset_cfg['percnt_lbl']

        self.files = []
        self._set_files()
        super(SemiDataset, self).__init__()

    def _set_files(self):
        if self.split == "val":
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        elif self.split == "test":
            file_list = os.path.join(self.root, 'list', f"{self.split}" + ".txt")
        elif self.split in ["train_supervised", "train_unsupervised"]:
            file_list = os.path.join(self.root, 'list', f"{self.percnt_lbl}_{self.split}" + ".txt")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        img_name_list = np.loadtxt(file_list, dtype=str)
        if img_name_list.ndim == 2:
            return img_name_list[:, 0]

        self.files = img_name_list
    
  
    def __getitem__(self, index):
        image_A_path    = os.path.join(self.root, 'A', self.files[index])
        image_B_path    = os.path.join(self.root, 'B', self.files[index])
        image_A         = self.read(image_A_path)
        image_B         = self.read(image_B_path)
                
        label_path  = os.path.join(self.root, 'label', self.files[index])
        label = np.asarray(self.read(label_path), dtype=np.int32)

        image_A = transforms.ToTensor()(image_A)
        image_B = transforms.ToTensor()(image_B)
        label[label>=1] = 1
        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()

        return image_A, image_B, label, self.files[index]
        
    
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
        return len(self.files)
