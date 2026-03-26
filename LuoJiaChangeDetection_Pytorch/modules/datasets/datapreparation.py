import numpy as np
import os
from PIL import Image

def start_points(size, split_size, overlap=0):
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


def findthepatch(overlap = 0,
    split_width = 128,
    split_height = 128,
    original_name='', index=0):

  
    img = np.asarray(Image.open(original_name))
    img_h, img_w, _ = img.shape
    X_points = start_points(img_w, split_width, overlap)
    Y_points = start_points(img_h, split_height, overlap)
    
    ind = 0
    for i in Y_points:
        for j in X_points:  
            split = img[i:i + split_height, j:j + split_width,:]
            if(index==ind):
                return split   
            ind += 1
    


def clipimages(overlap = 0,
    split_width = 128,
    split_height = 128,
    inputpath = None):
    splits = []
    count = 0
    img_list = os.listdir(inputpath)
    for img_name in img_list:
      
        img = os.path.join(inputpath,img_name)
        img = np.asarray(Image.open(img))
        img_h, img_w, _ = img.shape
        X_points = start_points(img_w, split_width, overlap)
        Y_points = start_points(img_h, split_height, overlap)
        for i in Y_points:
            for j in X_points:  
                split = img[i:i + split_height, j:j + split_width,:]

                splits.append(split)
                count += 1
    return splits, count

def pathcounting(overlap = 0,
    split_width = 128,
    split_height = 128,
    inputpath = None):
    patch_list = []
    count = 0
    img_list = os.listdir(inputpath)
    for img_name in img_list:
        
        img = os.path.join(inputpath,img_name)
        img = np.asarray(Image.open(img))
        img_h, img_w, _ = img.shape
        X_points = start_points(img_w, split_width, overlap)
        Y_points = start_points(img_h, split_height, overlap)
        
        # count += len(X_points)*len(Y_points)
        
        # id=0
        for i in Y_points:
            for j in X_points:  
                patch_name =os.path.join(
                   inputpath, os.path.splitext(img_name)[0]+ '_'+str(id)+os.path.splitext(img_name)[1]) 
                # print(patch_name)
                # count += 1
                patch_list.append(patch_name)
                # patch_list[count] = patch_name
                # id += 1
    return  patch_list

patch_list = pathcounting(overlap = 0,
    split_width = 128,
    split_height = 128,
    inputpath = '/media/lscsc/nas/xianping/ZXK/CDDemo/CDFramework/data/dataset1/train/A')
pickup = patch_list[42]
original_name = pickup[:pickup.rfind('_')]+os.path.splitext(pickup)[1]
ind = os.path.splitext(pickup)[0][pickup.rfind('_')+1:]

# print(pickup)
# print(original_name)
# print(ind)

thepatch = findthepatch(original_name = original_name,index = int(ind))
# print(np.shape(thepatch))
