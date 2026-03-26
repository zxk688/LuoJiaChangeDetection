from modules import cfg_from_yaml_file
from frameworks import BuildFramework
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='./configs/a2net/a2net_dsifn.yml',required=False, help="Path of the config file")
    
    opt = parser.parse_args()
    cfg = cfg_from_yaml_file(opt.config)
    
    CD_framework = BuildFramework(configs = cfg)
    if cfg["mode"]=="train":
        CD_framework.training()
    elif cfg["mode"]=="test":
        CD_framework.testing()






