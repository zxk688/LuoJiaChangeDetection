import os
import cv2
import torch
import numpy as np
import collections
import scipy.io as sio
from PIL import Image
from .pipelines import Evaluation, BuildDataTransform, DataTransformBuilder, Compose


'''BaseDataset'''
class BaseChangeDetectionDataset(torch.utils.data.Dataset):
    def __init__(self, mode, logger_handle, dataset_cfg):
        self.mode = mode
        self.logger_handle = logger_handle
        self.dataset_cfg = dataset_cfg   

        if 'data_pipelines' in self.dataset_cfg.keys():
            self.transforms = self.constructtransforms(self.dataset_cfg['data_pipelines'])
            
    '''constructtransforms'''
    def constructtransforms(self, data_pipelines):
        transforms = []
        for data_pipeline in data_pipelines:
            if isinstance(data_pipeline, collections.abc.Sequence):
                assert len(data_pipeline) == 2
                assert isinstance(data_pipeline[1], dict)
                transform_type, transform_cfg = data_pipeline
                transform_cfg['type'] = transform_type
                transform = BuildDataTransform(transform_cfg)
            else:
                assert isinstance(data_pipeline, dict)
                transform = BuildDataTransform(data_pipeline)
            transforms.append(transform)
        transforms = Compose(transforms)
        # return
        return transforms
    
    '''synctransforms'''
    def synctransforms(self, sample_meta):
        if self.mode =='train':
            sample_meta = self.transforms(sample_meta)
        else:
            if self.mode == 'test':
                cd_target = sample_meta.pop('gt')
            sample_meta = self.transforms(sample_meta)
            if self.mode == 'test':
                sample_meta['seg_target'] = cd_target
        return sample_meta
    
    '''randompalette'''
    @staticmethod
    def randompalette(num_classes):
        palette = [0] * (num_classes * 3)
        for j in range(0, num_classes):
            i, lab = 0, j
            palette[j * 3 + 0], palette[j * 3 + 1], palette[j * 3 + 2] = 0, 0, 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        palette = np.array(palette).reshape(-1, 3)
        palette = palette.tolist()
        return palette