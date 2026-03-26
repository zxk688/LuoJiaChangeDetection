from modules import cfg_from_yaml_file
from frameworks import BuildFramework
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',type=str,default='configs/bit/bit_levir.yml',required=False, help="Path of the config file")
    
    opt = parser.parse_args()
    cfg = cfg_from_yaml_file(opt.config)
    
    CD_framework = BuildFramework(configs = cfg)
    if cfg["mode"]=="train":
        CD_framework.training()
    elif cfg["mode"]=="test":
        CD_framework.testing()

#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/fcnpp/fcnpp_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/fdcnn/fdcnn_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/hanet/hanet_levir_train.yml 
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/snunet/snunet_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/rdpnet/rdpnet_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/tinycd/tinycd_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/ussfcnet/ussfcnet_levir_train.yml 
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/fc_siam/fc_siamconc_levir.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/fc_ef/fc_ef_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/dsifn/dsifn_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/dminet/dminet_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/changeformer/changeformer_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection-master/configs/a2net/a2net_levir_train.yml
#? /root/autodl-tmp/LuoJiaChangeDetection_mindspore/configs/icifnet/icignet.yml
#? /root/autodl-tmp/LuoJiaChangeDetection_mindspore/configs/dsamnet/dsamnet_levir.yml
#? /root/autodl-tmp/LuoJiaChangeDetection_mindspore/configs/bit/bit_levir.yml

