import luojianet
import luojianet.nn as nn
import luojianet.dataset as ds
from luojianet import load_param_into_net,load_checkpoint

import copy
import numpy as np
import time
import cv2
import os
from modules import (
    Logger,touchdir, touchfile, BuildDataset, BuildLoss,BuildOptimizer,BuildScheduler, BuildChangeDetector, metrics)

import luojianet
import luojianet.nn as nn
import luojianet.ops as ops
from luojianet import load_param_into_net,load_checkpoint
from luojianet.train.callback import LossMonitor, TimeMonitor, ModelCheckpoint, CheckpointConfig
from luojianet.train.loss_scale_manager import FixedLossScaleManager
from luojianet import nn, ops, Parameter, Tensor, context, Model
from luojianet.communication import init
import luojianet.dataset as ds
from luojianet.nn import SGD,Adam
from luojianet.train.loss_scale_manager import FixedLossScaleManager
from luojianet.dataset import ImageFolderDataset
from luojianet.train.callback import Callback
from luojianet import save_checkpoint
import luojianet.ops as F
from luojianet.train.serialization import load_checkpoint, load_param_into_net
import luojianet.ops.operations as P
import os
import cv2



def evaluate_model(net, eval_data, output_path=None):
    """Inference function."""
    net.set_train(False)
    all_preds = []
    all_gts = []
    count = 0
    for item in eval_data:
        images = item[0]
        labels = item[1]
        fname = item[2].asnumpy()
        labels_A,labels_B, _ = labels[:,0:7],labels[:,7:14],labels[:,-1]
        outputs_A, outputs_B, out_change = net(images)
        
        change_mask = F.sigmoid(out_change)
        change_mask = (change_mask>0.5).squeeze(1).long()
        
        preds_A = F.argmax(outputs_A, dim=1)
        preds_B = F.argmax(outputs_B, dim=1)   
        # if output_path!=None:
        #     result1 = preds_A.squeeze(1).transpose(1, 2, 0).astype(np.uint8)
        #     for i in range(result1.shape[-1]):
        #         cv2.imwrite(os.path.join(output_path, fname[i]), (result1[:,:,i])*255)
        #     result2 = preds_B.squeeze(1).transpose(1, 2, 0).astype(np.uint8)
        #     for j in range(result2.shape[-1]):
        #         cv2.imwrite(os.path.join(output_path, fname[j]), (result2[:,:,j])*255)
         
        preds_A = (preds_A*change_mask).asnumpy().squeeze()
        preds_B = (preds_B*change_mask).asnumpy().squeeze()
        
        labels_A = F.argmax(labels_A, dim=1).asnumpy().squeeze()
        labels_B = F.argmax(labels_B, dim=1).asnumpy().squeeze()


        for (pred_A, pred_B, label_A, label_B) in zip(preds_A, preds_B, labels_A, labels_B):
            all_preds.append(pred_A)
            all_preds.append(pred_B)
            all_gts.append(label_A)
            all_gts.append(label_B)

    accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
                np.concatenate([p.ravel() for p in all_gts]),label_values=['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field'])
    return accuracy


class EvalCallback(Callback):
    """Callback for inference while training."""
    def __init__(self, network, eval_data, model_name, num_classes, snapshot_path, eval_interval=1, device_id=0):
        self.network = network
        self.eval_data = eval_data
        self.best_iouarray = None
        self.best_miou = 0
        self.best_epoch = 0
        self.num_classes = num_classes
        self.eval_interval = eval_interval
        self.device_id = device_id
        self.snapshot_path = snapshot_path
        self.model_name = model_name

    def epoch_end(self, run_context):
        """Executions after each epoch."""
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num

        if cur_epoch % self.eval_interval == 0:
            miou = evaluate_model(self.network, self.eval_data)
            print(f'Evaluation completed! The test mIoU is {miou}.')  
            if miou > self.best_miou:
                self.best_miou = miou
                self.best_epoch = cur_epoch
                save_checkpoint(self.network, self.snapshot_path + self.model_name + "_best.ckpt")
            
            self.network.set_train(True)
            

def eval_callback(network, CD_dataset_eval, model_name, num_classes, snapshot_path, eval_interval, device_id=0):
    """Create an object for inference while training."""
    eval_cb = EvalCallback(network=network,
                        eval_data=CD_dataset_eval,
                        model_name=model_name,
                        num_classes=num_classes,
                        snapshot_path=snapshot_path,
                        eval_interval=eval_interval,
                        device_id=device_id)
    return eval_cb

class SemanticChangeDetectionFramework():
    def __init__(self,configs):
        super().__init__()
        self.configs=configs
        self._get_CD_dataest()
        self._get_device()
        self._get_CD_model_using_name()
        self._get_logger_handle()

    def training(self):
        context.set_context(device_target='Ascend')
        self._get_lr_scheduler()
        self._get_optim()
        self._get_loss()
        self.device_id = 0
        
        steps_per_epoch = self.CD_dataset_train.get_dataset_size()
        time_cb = TimeMonitor(data_size=steps_per_epoch)
        loss_cb = LossMonitor(per_print_times=steps_per_epoch)
        self.net.set_train(True)
        model = Model(self.net, loss_fn=self.loss, optimizer=self.optimizer, metrics={'accuracy'})
        ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch * 10,
                                   keep_checkpoint_max=10)
        ckpt_cb = ModelCheckpoint(prefix='{}'.format(self.configs['model_name']),
                              directory=self.configs["train"]["snapshots_dir"],
                              config=ckpt_config)
        cb = [time_cb, loss_cb, ckpt_cb]
        eval_cb = eval_callback(network=self.net,
                        CD_dataset_eval=self.CD_dataset_eval,
                        model_name=self.configs['model_name'],
                        num_classes=self.configs['model']['output_channels'],
                        snapshot_path=self.configs["train"]["snapshots_dir"],
                        eval_interval=self.configs["train"]["save_intervals"],
                        device_id=self.device_id)
        cb.append(eval_cb)
        self.logger_handle.info("============== Starting Training ==============")
        model.train(self.configs["train"]["epochs"], self.CD_dataset_train, callbacks=cb)
        self.logger_handle.info("============== Training Finished==============")
        
    def testing(self):
        touchdir(self.configs["test"]["save_path"])
        self.logger_handle.info("============== Starting Testing ==============")
        acc = evaluate_model(self.net, self.CD_dataset_test, self.configs["test"]["save_path"])
        self.logger_handle.info(f'Test mIoU: {acc}')
        self.logger_handle.info("============== Testing Finished==============")
             

    def _get_CD_model_using_name(self):
        self.net = BuildChangeDetector(changedetector_cfg=copy.deepcopy(self.configs['model']))  
        if self.configs["mode"]=="test":
            load_param_into_net(self.net, load_checkpoint(self.configs["test"]["checkpoint"]))
       
   
    def _get_CD_dataest(self):
        if self.configs["mode"]=="train":
            CD_dataset_train = BuildDataset(mode='train',logger_handle=None,dataset_cfg=self.configs['dataset'])
            CD_dataset_train = ds.GeneratorDataset(source=CD_dataset_train,column_names=['image','label'],shuffle=True)
            self.CD_dataset_train = CD_dataset_train.batch(batch_size=self.configs["train"]["batch_size"])
            if self.configs["eval"]:
                CD_dataset_eval = BuildDataset(mode='val',logger_handle=None,dataset_cfg=self.configs['dataset'])
                CD_dataset_eval=ds.GeneratorDataset(source=CD_dataset_eval,column_names=['image','label','fname'],shuffle=False)
                self.CD_dataset_eval = CD_dataset_eval.batch(self.configs["eval"]["batch_size"], drop_remainder=False)
        elif self.configs["mode"]=="test":
            CD_dataset_test = BuildDataset(mode='test',logger_handle=None,dataset_cfg=self.configs['dataset'])
            CD_dataset_test = ds.GeneratorDataset(source=CD_dataset_test,column_names=['image','label','fname'],shuffle=False)
            self.CD_dataset_test = CD_dataset_test.batch(self.configs["test"]["batch_size"], drop_remainder=False)
            
            
    def _get_optim(self):  
        self.optimizer = BuildOptimizer(self.net, self.scheduler, self.configs['optimizer'])

    def _get_lr_scheduler(self):
        if 'scheduler' in self.configs:
            self.scheduler = BuildScheduler(scheduler_cfg=self.configs['scheduler'])
        else:
            self.scheduler = None
            
    def _get_loss(self):   
        self.loss = BuildLoss(loss_cfg=self.configs['losses']['loss_cls'])
        
    def _get_device(self):
        self.device = self.configs["device"]

    def _get_logger_handle(self):        
        touchfile(self.configs['logfilepath'])
        self.logger_handle = Logger(self.configs['logfilepath'])
