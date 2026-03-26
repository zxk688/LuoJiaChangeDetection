import os

'''touchdir'''
def touchdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        for i in os.listdir(path):
            os.remove(os.path.join(path,i))

'''touchfile'''
def touchfile(file_path):
    outputpath = os.path.split(file_path)[0]
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

