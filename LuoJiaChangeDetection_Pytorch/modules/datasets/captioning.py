import torch
from torch.utils.data import Dataset
import h5py
import json
import os
from torchvision import transforms  


class CaptionDataset(Dataset):
    
    def __init__(self, mode, logger_handle, dataset_cfg):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param mode: mode, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        transform=transforms.Compose([normalize])
        self.mode = mode
        
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        data_folder = self.dataset_cfg['rootdir']
        data_name = self.dataset_cfg['data_name']
        
        # Open hdf5 file where images are stored

        if os.path.exists(os.path.join(data_folder, self.mode + '_IMAGES_' + data_name + '.hdf5')):
            self.h = h5py.File(os.path.join(data_folder, self.mode + '_IMAGES_' + data_name + '.hdf5'), 'r')
        self.imgs = self.h['images']

        # Captions per image
        self.cpi = self.h.attrs['captions_per_image']

        # Load encoded captions (completely into memory)
        with open(os.path.join(data_folder, self.mode + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.mode + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform #None

        # Total number of datapoints
        # #FIXME：original
        self.dataset_size = int(len(self.captions) / 1)

        if self.mode == 'test':
            # Load word map (word2ix)
            word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')
            with open(word_map_file, 'r') as f:
                self.word_map = json.load(f)


    def __getitem__(self, i):
        # FIXME：original
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        img = torch.FloatTensor(self.imgs[i // self.cpi] / 255.)
        if self.transform is not None:
            if img.shape == torch.Size([3,256,256]):
                img = self.transform(img)
            elif img.shape == torch.Size([2,3,256,256]):

                img[0] = self.transform(img[0])
                img[1] = self.transform(img[1])

        caption = torch.LongTensor(self.captions[i])
        caplen = torch.LongTensor([self.caplens[i]])

        if self.mode == 'train':
            return img, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(
                self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
