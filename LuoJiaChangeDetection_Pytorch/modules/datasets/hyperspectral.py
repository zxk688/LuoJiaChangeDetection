
import numpy as np
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.io import loadmat

import torch
from torch.utils import data

def chooose_train_and_test_point(train_data, test_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}

    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class
    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    return total_pos_train, total_pos_test, number_train, number_test
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    #左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    # print("**************************************************")
    # print("patch is : {}".format(patch))
    # print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    # print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image

def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=float)
    # 中心区域
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape
    #左边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]
    #右边镜像
    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, patch=5, band_patch=3):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=float)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=float)
    # x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=float)
    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    # print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    # print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    # print("**************************************************")

    x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
    x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
    # print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    # print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    # print("**************************************************")
    return x_train_band, x_test_band
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, num_classes):
    y_train = []
    y_test = []
    # y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)

    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    # print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    # print("**************************************************")
    return y_train, y_test
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()


def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()



class HyperspectralDataset():
    def __init__(self, mode, logger_handle, dataset_cfg):
        super(HyperspectralDataset, self).__init__()
        self.mode = mode
        self.dataset_cfg = dataset_cfg
        self.logger_handle = logger_handle
        
        self.patch = self.dataset_cfg['patches']
        self.band_patches = self.dataset_cfg['band_patches']
        self.train_number=self.dataset_cfg['train_number']

    def __getdataset__(self):
        data_t1 = loadmat(self.dataset_cfg["rootdir"][0])[self.dataset_cfg["key_names"][0]]
        data_t2 = loadmat(self.dataset_cfg["rootdir"][1])[self.dataset_cfg["key_names"][1]]
        data_label = loadmat(self.dataset_cfg["rootdir"][2])[self.dataset_cfg["key_names"][2]]
        uc_position = np.array(np.where(data_label==0)).transpose(1,0)
        c_position = np.array(np.where(data_label==255)).transpose(1,0)
        # print((uc_position.shape[0],c_position.shape[0]))
        data_label = (data_label-data_label.min())/(data_label.max()-data_label.min())
        data_label[data_label==0]=2

        selected_uc = np.random.choice(uc_position.shape[0], int(self.train_number), replace = False)
        selected_c = np.random.choice(c_position.shape[0], int(self.train_number), replace = False)
        selected_uc_position=uc_position[selected_uc]
        selected_c_position=c_position[selected_c]
        TR = np.zeros(data_label.shape)
        for i in range (int(self.train_number)):
            TR[selected_c_position[i][0],selected_c_position[i][1]]=1
            TR[selected_uc_position[i][0],selected_uc_position[i][1]]=2
        TE=data_label-TR

        num_classes = np.max(TR)
        num_classes=int(num_classes)
        input1_normalize = np.zeros(data_t1.shape)
        input2_normalize = np.zeros(data_t1.shape)
        for i in range(data_t1.shape[2]):
            input_max = max(np.max(data_t1[:,:,i]),np.max(data_t2[:,:,i]))
            input_min = min(np.min(data_t1[:,:,i]),np.min(data_t2[:,:,i]))
            input1_normalize[:,:,i] = (data_t1[:,:,i]-input_min)/(input_max-input_min)
            input2_normalize[:,:,i] = (data_t2[:,:,i]-input_min)/(input_max-input_min)

        self.height, self.width, self.band = data_t1.shape
        print("height={0}, width={1}, band={2}".format(self.height, self.width, self.band))
        if self.mode=="train":
            total_pos_train, total_pos_test, number_train, number_test = chooose_train_and_test_point(TR, TE, num_classes)
            mirror_image_t1 = mirror_hsi(self.height, self.width, self.band, input1_normalize, patch=self.patch)
            mirror_image_t2 = mirror_hsi(self.height, self.width, self.band, input2_normalize, patch=self.patch)
            x_train_band_t1, x_test_band_t1 = train_and_test_data(mirror_image_t1, self.band, total_pos_train, total_pos_test, 
                                                                patch=self.patch, band_patch=self.band_patches)
            x_train_band_t2, x_test_band_t2 = train_and_test_data(mirror_image_t2, self.band, total_pos_train, total_pos_test, 
                                                                patch=self.patch, band_patch=self.band_patches)
            y_train, y_test = train_and_test_label(number_train, number_test, num_classes)
            #-------------------------------------------------------------------------------
            # load data
            x_train_t1=torch.from_numpy(x_train_band_t1.transpose(0,2,1)).type(torch.FloatTensor) 
            x_train_t2=torch.from_numpy(x_train_band_t2.transpose(0,2,1)).type(torch.FloatTensor) 
            y_train=torch.from_numpy(y_train).type(torch.LongTensor) #[695]
            Label_train=data.TensorDataset(x_train_t1,x_train_t2,y_train)
            x_test_t1=torch.from_numpy(x_test_band_t1.transpose(0,2,1)).type(torch.FloatTensor)
            x_test_t2=torch.from_numpy(x_test_band_t2.transpose(0,2,1)).type(torch.FloatTensor) 
            y_test=torch.from_numpy(y_test).type(torch.LongTensor) # [9671]
            Label_test=data.TensorDataset(x_test_t1,x_test_t2,y_test)

            return Label_train, Label_test

        elif self.mode=="test":
            
            mirror_image_t1 = mirror_hsi(self.height, self.width, self.band, input1_normalize, patch=self.patch)
            mirror_image_t2 = mirror_hsi(self.height, self.width, self.band, input2_normalize, patch=self.patch)
            x1_true = np.zeros((self.height*self.width, self.patch, self.patch, self.band), dtype=float)
            x2_true = np.zeros((self.height*self.width, self.patch, self.patch, self.band), dtype=float)
            y_true=[]
            for i in range(self.height):
                for j in range(self.width):
                    x1_true[i*self.width+j,:,:,:]=mirror_image_t1[i:(i+self.patch),j:(j+self.patch),:]
                    x2_true[i*self.width+j,:,:,:]=mirror_image_t2[i:(i+self.patch),j:(j+self.patch),:]
                    y_true.append(i)
            y_true = np.array(y_true)
            x1_true_band = gain_neighborhood_band(x1_true, self.band, self.band_patches, self.patch)
            x2_true_band = gain_neighborhood_band(x2_true, self.band, self.band_patches, self.patch)
            x1_true_band=torch.from_numpy(x1_true_band.transpose(0,2,1)).type(torch.FloatTensor)
            x2_true_band=torch.from_numpy(x2_true_band.transpose(0,2,1)).type(torch.FloatTensor)
            y_true=torch.from_numpy(y_true).type(torch.LongTensor)
            Label_true=data.TensorDataset(x1_true_band,x2_true_band,y_true)
            return Label_true
        


  