# import torch
# from torch.utils import data

# import copy
# import numpy as np
# import time
# import cv2
# import os
# from modules import (
#     Logger,touchdir, touchfile, BuildDataset, BuildChangeDetector, BuildOptimizer, BuildScheduler,BuildLoss, metrics)


# class BinaryChangeDetectionFramework():
#     def __init__(self,configs):
#         super().__init__()
#         self.configs=configs
#         self._get_CD_dataloader()
#         self._get_device()
#         self._get_CD_model_using_name()
#         self._get_logger_handle()
        
#     def training(self):
#         self.logger_handle.info('Training')
#         self._get_optim()
#         self._get_lr_scheduler()
#         self._get_loss()
        
#         for epoch in range(self.configs["train"]["epochs"]):
#             loss_list=[]
#             self.CD_model.train()
#             for i,batch in enumerate(self.CD_dataloader_train):
#                 self.optimizer.zero_grad()
#                 pre_tensor, post_tensor, label_tensor, fname = batch["image1"], batch["image2"], batch["gt"], batch["fname"]
#                 pre_tensor = pre_tensor.to(self.device)
#                 post_tensor = post_tensor.to(self.device)
#                 if self.configs['model']['output_channels'] < 2:
#                     label_tensor = (label_tensor > 0).float().to(self.device) 
#                 else:
#                     label_tensor = label_tensor.type(torch.LongTensor).to(self.device)
                    
#                 prediction = self.CD_model(pre_tensor, post_tensor)
    
#                 if self.configs['model']['output_channels'] < 2:
#                     total_loss = self.loss(prediction,label_tensor)
#                 else:
#                     total_loss = self.loss(prediction,(label_tensor.squeeze(1)))
                
#                 loss_list.append(total_loss.item()) 
#                 total_loss.backward()
#                 self.optimizer.step()
#             torch.cuda.empty_cache() 
#             loss_avg=sum(loss_list)/len(loss_list)
#             self.scheduler.step(loss_avg) 
#             lr=self.optimizer.param_groups[0]['lr']
#             self.logger_handle.info(time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time()))+f', epoch={epoch+1} | loss={loss_avg:.7f} | lr={lr:.7f}')
#             if (epoch+1)%self.configs["train"]["save_intervals"]==0:
#                 self.current_epoch=epoch+1
#                 self._save_model()
#                 if self.configs["eval"]:
#                     self.eval()


#     def eval(self):
#         self.CD_model.eval()      
#         all_preds = []
#         all_gts = []
            
#         # for _, batch in enumerate(self.CD_dataloader_eval):
#         #     pre_tensor, post_tensor, label_tensor, fname = batch["image1"], batch["image2"], batch["gt"], batch["fname"]
#         #     pre_tensor = pre_tensor.to(self.device)
#         #     post_tensor = post_tensor.to(self.device)
#         #     label_tensor = label_tensor.to(self.device)
#         #     probs = self.CD_model(pre_tensor, post_tensor)

#         #     if self.configs['model']['output_channels'] < 2:
#         #         prediction = torch.where(probs>0.5,1,0)
#         #     else:
#         #         prediction = torch.argmax(probs, dim=1)

#         #     all_preds.append(prediction.detach().cpu().numpy())
#         #     all_gts.append(label_tensor.detach().cpu().numpy())
#         # accuracy = metrics(np.concatenate([p.ravel() for p in all_preds]),
#         #             np.concatenate([p.ravel() for p in all_gts]).ravel())
#         for _, batch in enumerate(self.CD_dataloader_eval):
#             pre_tensor, post_tensor, label_tensor, fname = (
#                 batch["image1"].to(self.device),
#                 batch["image2"].to(self.device),
#                 batch["gt"].to(self.device),
#                 batch["fname"],
#             )
            
#             probs = self.CD_model(pre_tensor, post_tensor)
        

#             if self.configs['model']['output_channels'] < 2:     # 二分类
#                 prediction = (probs > 0.5).to(torch.uint8)
#                 label_tensor = (label_tensor > 0).to(torch.uint8)   # 把 0/255 → 0/1
#             else:                                                # 多分类
#                 prediction = torch.argmax(probs, dim=1).to(torch.uint8)
#                 label_tensor = label_tensor.to(torch.uint8)

#             all_preds.append(prediction.cpu().numpy())           # 已经是 uint8
#             all_gts.append(label_tensor.cpu().numpy())

#         accuracy = metrics(
#             np.concatenate([p.ravel() for p in all_preds]),
#             np.concatenate([g.ravel() for g in all_gts]),
#             logger_handle=self.logger_handle
#         )

        
#         torch.cuda.empty_cache()    
#         self.logger_handle.info(f'Epoch {self.current_epoch} evaluation completed, the accuracy is {accuracy}')

    

                
#     def testing(self):
#         self.logger_handle.info("testing")
#         self.CD_model.eval()
#         touchdir(self.configs["test"]["save_path"])

#         all_preds, all_gts = [], []

#         for _, data in enumerate(self.CD_dataloader_test):
#             # ──────────────────── 数据搬到 GPU ────────────────────
#             pre_tensor  = data["image1"].to(self.device)
#             post_tensor = data["image2"].to(self.device)
#             label_tensor = data["gt"].to(self.device)          # 原始 GT 可能是 0/255

#             # ──────────────────── 前向推理 ────────────────────
#             logits = self.CD_model(pre_tensor, post_tensor)
#             probs  = logits

#             # ──────────────────── 二值 / 多类分支 ────────────────────
#             if self.configs['model']['output_channels'] < 2:   # 二分类
#                 prediction   = (probs > 0.5).to(torch.uint8)   # ← 0/1, uint8
#                 label_tensor = (label_tensor > 0).to(torch.uint8)
#             else:                                              # 多分类（N 通道 softmax 已经在模型里）
#                 prediction   = torch.argmax(probs, dim=1).to(torch.uint8)
#                 label_tensor = label_tensor.to(torch.uint8)

#             # ──────────────────── 保存可视化 ────────────────────
#                 # ──────────────────── 保存可视化 ────────────────────
#             result = prediction.cpu().numpy()          # shape: [B, 1, H, W] 或者 [B, H, W]
#             # 如果多了 channel 维，就挤掉
#             if result.ndim == 4 and result.shape[1] == 1:
#                 result = result.squeeze(1)             # 变成 [B, H, W]
#             for j in range(result.shape[0]):
#                 mask = result[j]                       # shape: [H, W]
#                 # 0/1 → 0/255，转成 uint8
#                 mask = (mask * 255).astype(np.uint8)
#                 save_name = os.path.join(self.configs["test"]["save_path"], data["fname"][j])
#                 cv2.imwrite(save_name, mask)
#                 self.logger_handle.info(f"Saved: {save_name}")


#             # ──────────────────── 收集统计 ────────────────────
#             all_preds.append(result)                           # 已经 squeeze 过
#             all_gts.append(label_tensor.cpu().numpy())

#         # ──────────────────── 计算指标 ────────────────────
#         y_pred = np.concatenate([p.ravel() for p in all_preds]).astype(np.uint8)
#         y_true = np.concatenate([g.ravel() for g in all_gts]).astype(np.uint8)
#         accuracy = metrics(y_pred, y_true, logger_handle=self.logger_handle)

#         self.logger_handle.info(f"Test done – overall accuracy: {accuracy:.4f}")
#         torch.cuda.empty_cache()

#     def _get_CD_model_using_name(self):
        
#         self.changedecetor = BuildChangeDetector(changedetector_cfg=copy.deepcopy(self.configs['model']))        
                      
#         self.CD_model=self.changedecetor.to(self.device)

#         if self.configs["mode"]=="test":
#             self.CD_model.load_state_dict(torch.load(self.configs["test"]["checkpoint"]))

#     def _get_CD_dataloader(self):
#         if self.configs["mode"]=="train":
            
#             CD_dataset_train = BuildDataset(mode='train',logger_handle=None,dataset_cfg=self.configs['dataset'])
#             self.CD_dataloader_train=data.DataLoader(dataset=CD_dataset_train,batch_size=self.configs["train"]["batch_size"],shuffle=True,num_workers=0,pin_memory=False)
#             if self.configs["eval"]:
#                 CD_dataset_eval = BuildDataset(mode='val',logger_handle=None,dataset_cfg=self.configs['dataset'])
#                 self.CD_dataloader_eval=data.DataLoader(dataset=CD_dataset_eval,batch_size=self.configs["eval"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False)

#         elif self.configs["mode"]=="test":
#             CD_dataset_test = BuildDataset(mode='test',logger_handle=None,dataset_cfg=self.configs['dataset'])
#             self.CD_dataloader_test=data.DataLoader(dataset=CD_dataset_test,batch_size=self.configs["test"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False)

#     def _get_optim(self):    
#         self.optimizer = BuildOptimizer(self.changedecetor, self.configs['scheduler']['optimizer'])

#     def _get_lr_scheduler(self):
#         self.scheduler = BuildScheduler(optimizer=self.optimizer, scheduler_cfg=self.configs['scheduler'])

#     def _get_loss(self):
#         self.loss = BuildLoss(loss_cfg=self.configs['losses']['loss_cls'])
        
#     def _get_device(self):
#         self.device = self.configs["device"]
    
#     def _save_model(self):
#         snapshot_dir_full=os.path.join(self.configs["train"]["snapshots_dir"],self.configs["model_name"])
#         os.makedirs(snapshot_dir_full,exist_ok=True)
#         torch.save(self.CD_model.state_dict(),os.path.join(snapshot_dir_full,f'{self.current_epoch}.pth'))

#     def _get_logger_handle(self):        
#         touchfile(self.configs['logfilepath'])
#         self.logger_handle = Logger(self.configs['logfilepath'])


# import torch
# from torch.utils import data

# import copy
# import numpy as np
# import time
# import cv2
# import os
# from modules import (
#     Logger, touchdir, touchfile, BuildDataset, BuildChangeDetector, BuildOptimizer, BuildScheduler, BuildLoss, metrics)


# class BinaryChangeDetectionFramework():
#     def __init__(self, configs):
#         super().__init__()
#         self.configs = configs
#         self._get_CD_dataloader()
#         self._get_device()
#         self._get_CD_model_using_name()
#         self._get_logger_handle()
        
#         # [新增] 初始化最佳 IoU 记录
#         self.best_iou = 0.0

#     def training(self):
#         self.logger_handle.info('Training')
#         self._get_optim()
#         self._get_lr_scheduler()
#         self._get_loss()

#         for epoch in range(self.configs["train"]["epochs"]):
#             loss_list = []
#             self.CD_model.train()
#             for i, batch in enumerate(self.CD_dataloader_train):
#                 self.optimizer.zero_grad()
#                 pre_tensor, post_tensor, label_tensor, fname = batch["image1"], batch["image2"], batch["gt"], batch["fname"]
#                 pre_tensor = pre_tensor.to(self.device)
#                 post_tensor = post_tensor.to(self.device)
                
#                 if self.configs['model']['output_channels'] < 2:
#                     label_tensor = (label_tensor > 0).float().to(self.device)
#                 else:
#                     label_tensor = label_tensor.type(torch.LongTensor).to(self.device)

#                 prediction = self.CD_model(pre_tensor, post_tensor)

#                 if self.configs['model']['output_channels'] < 2:
#                     total_loss = self.loss(prediction, label_tensor)
#                 else:
#                     total_loss = self.loss(prediction, (label_tensor.squeeze(1)))

#                 loss_list.append(total_loss.item())
#                 total_loss.backward()
#                 self.optimizer.step()
            
#             torch.cuda.empty_cache()
#             loss_avg = sum(loss_list) / len(loss_list)
#             self.scheduler.step(loss_avg)
#             lr = self.optimizer.param_groups[0]['lr']
#             self.logger_handle.info(time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())) +
#                                     f', epoch={epoch+1} | loss={loss_avg:.7f} | lr={lr:.7f}')
            
#             if (epoch + 1) % self.configs["train"]["save_intervals"] == 0:
#                 self.current_epoch = epoch + 1
#                 self._save_model(is_best=False) # 保存常规 checkpoint
#                 if self.configs["eval"]:
#                     self.eval()

#     def eval(self):
#         self.CD_model.eval()
#         all_preds = []
#         all_gts = []

#         with torch.no_grad(): # 加上 no_grad 节省显存
#             for _, batch in enumerate(self.CD_dataloader_eval):
#                 pre_tensor, post_tensor, label_tensor, fname = (
#                     batch["image1"].to(self.device),
#                     batch["image2"].to(self.device),
#                     batch["gt"].to(self.device),
#                     batch["fname"],
#                 )

#                 probs = self.CD_model(pre_tensor, post_tensor)

#                 if self.configs['model']['output_channels'] < 2:     # 二分类
#                     prediction = (probs > 0.5).to(torch.uint8)
#                     label_tensor = (label_tensor > 0).to(torch.uint8)
#                 else:                                                # 多分类
#                     prediction = torch.argmax(probs, dim=1).to(torch.uint8)
#                     label_tensor = label_tensor.to(torch.uint8)

#                 all_preds.append(prediction.cpu().numpy())
#                 all_gts.append(label_tensor.cpu().numpy())

#         # 展平所有预测和标签进行计算
#         y_pred_flat = np.concatenate([p.ravel() for p in all_preds])
#         y_true_flat = np.concatenate([g.ravel() for g in all_gts])

#         # [修改] 计算 5 项指标
#         score_dict = self._calculate_metrics(y_pred_flat, y_true_flat)
        
#         # 格式化输出
#         log_str = (f"Epoch {self.current_epoch} Eval Metrics: "
#                    f"IoU={score_dict['iou']:.4f}, "
#                    f"F1={score_dict['f1']:.4f}, "
#                    f"Pre={score_dict['precision']:.4f}, "
#                    f"Rec={score_dict['recall']:.4f}, "
#                    f"Acc={score_dict['accuracy']:.4f}")
        
#         self.logger_handle.info(log_str)

#         # [新增] 记录并保存 IoU 最高的模型
#         if score_dict['iou'] > self.best_iou:
#             self.best_iou = score_dict['iou']
#             self.logger_handle.info(f"!!! NEW BEST IoU: {self.best_iou:.4f} (Epoch {self.current_epoch}) !!!")
#             self._save_model(is_best=True) # 保存最佳模型

#         torch.cuda.empty_cache()

#     def testing(self):
#         self.logger_handle.info("testing")
#         self.CD_model.eval()
#         touchdir(self.configs["test"]["save_path"])

#         all_preds, all_gts = [], []

#         with torch.no_grad():
#             for _, data in enumerate(self.CD_dataloader_test):
#                 pre_tensor = data["image1"].to(self.device)
#                 post_tensor = data["image2"].to(self.device)
#                 label_tensor = data["gt"].to(self.device)

#                 logits = self.CD_model(pre_tensor, post_tensor)
#                 probs = logits

#                 if self.configs['model']['output_channels'] < 2:
#                     prediction = (probs > 0.5).to(torch.uint8)
#                     label_tensor = (label_tensor > 0).to(torch.uint8)
#                 else:
#                     prediction = torch.argmax(probs, dim=1).to(torch.uint8)
#                     label_tensor = label_tensor.to(torch.uint8)

#                 # 保存图片逻辑
#                 result = prediction.cpu().numpy()
#                 if result.ndim == 4 and result.shape[1] == 1:
#                     result = result.squeeze(1)
                
#                 for j in range(result.shape[0]):
#                     mask = result[j]
#                     mask = (mask * 255).astype(np.uint8)
#                     save_name = os.path.join(self.configs["test"]["save_path"], data["fname"][j])
#                     cv2.imwrite(save_name, mask)
#                     # self.logger_handle.info(f"Saved: {save_name}") # 可以注释掉避免刷屏

#                 all_preds.append(result)
#                 all_gts.append(label_tensor.cpu().numpy())

#         # [修改] 计算并打印测试集指标
#         y_pred = np.concatenate([p.ravel() for p in all_preds]).astype(np.uint8)
#         y_true = np.concatenate([g.ravel() for g in all_gts]).astype(np.uint8)
        
#         score_dict = self._calculate_metrics(y_pred, y_true)
        
#         self.logger_handle.info("---------------- Test Results ----------------")
#         self.logger_handle.info(f"IoU      : {score_dict['iou']:.4f}")
#         self.logger_handle.info(f"F1-Score : {score_dict['f1']:.4f}")
#         self.logger_handle.info(f"Precision: {score_dict['precision']:.4f}")
#         self.logger_handle.info(f"Recall   : {score_dict['recall']:.4f}")
#         self.logger_handle.info(f"Accuracy : {score_dict['accuracy']:.4f}")
#         self.logger_handle.info("----------------------------------------------")
        
#         torch.cuda.empty_cache()

#     # [新增] 内部辅助函数：计算混淆矩阵和指标
#     def _calculate_metrics(self, pred, target):
#         """
#         输入: 展平的 numpy 数组 (0 或 1)
#         输出: 包含 5 项指标的字典
#         """
#         # 确保输入是二值的 (0/1)
#         pred = pred > 0
#         target = target > 0

#         tp = np.logical_and(pred, target).sum()
#         tn = np.logical_and(~pred, ~target).sum()
#         fp = np.logical_and(pred, ~target).sum()
#         fn = np.logical_and(~pred, target).sum()

#         eps = 1e-6 # 防止除零错误

#         accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
#         precision = tp / (tp + fp + eps)
#         recall = tp / (tp + fn + eps)
#         f1 = 2 * (precision * recall) / (precision + recall + eps)
#         iou = tp / (tp + fp + fn + eps)

#         return {
#             "iou": iou,
#             "precision": precision,
#             "recall": recall,
#             "f1": f1,
#             "accuracy": accuracy
#         }

#     def _get_CD_model_using_name(self):
#         self.changedecetor = BuildChangeDetector(changedetector_cfg=copy.deepcopy(self.configs['model']))
#         self.CD_model = self.changedecetor.to(self.device)

#         if self.configs["mode"] == "test":
#             self.CD_model.load_state_dict(torch.load(self.configs["test"]["checkpoint"]))

#     def _get_CD_dataloader(self):
#         if self.configs["mode"] == "train":
#             CD_dataset_train = BuildDataset(mode='train', logger_handle=None, dataset_cfg=self.configs['dataset'])
#             self.CD_dataloader_train = data.DataLoader(dataset=CD_dataset_train, batch_size=self.configs["train"]["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
#             if self.configs["eval"]:
#                 CD_dataset_eval = BuildDataset(mode='val', logger_handle=None, dataset_cfg=self.configs['dataset'])
#                 self.CD_dataloader_eval = data.DataLoader(dataset=CD_dataset_eval, batch_size=self.configs["eval"]["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

#         elif self.configs["mode"] == "test":
#             CD_dataset_test = BuildDataset(mode='test', logger_handle=None, dataset_cfg=self.configs['dataset'])
#             self.CD_dataloader_test = data.DataLoader(dataset=CD_dataset_test, batch_size=self.configs["test"]["batch_size"], shuffle=False, num_workers=0, pin_memory=False)

#     def _get_optim(self):
#         self.optimizer = BuildOptimizer(self.changedecetor, self.configs['scheduler']['optimizer'])

#     def _get_lr_scheduler(self):
#         self.scheduler = BuildScheduler(optimizer=self.optimizer, scheduler_cfg=self.configs['scheduler'])

#     def _get_loss(self):
#         self.loss = BuildLoss(loss_cfg=self.configs['losses']['loss_cls'])

#     def _get_device(self):
#         self.device = self.configs["device"]

#     def _save_model(self, is_best=False):
#         snapshot_dir_full = os.path.join(self.configs["train"]["snapshots_dir"], self.configs["model_name"])
#         os.makedirs(snapshot_dir_full, exist_ok=True)
        
#         # 保存当前 epoch 模型
#         if not is_best:
#             torch.save(self.CD_model.state_dict(), os.path.join(snapshot_dir_full, f'{self.current_epoch}.pth'))
        
#         # [新增] 单独保存最佳 IoU 模型
#         if is_best:
#             torch.save(self.CD_model.state_dict(), os.path.join(snapshot_dir_full, 'best_iou_model.pth'))
#             self.logger_handle.info(f"Saved best model to {os.path.join(snapshot_dir_full, 'best_iou_model.pth')}")

#     def _get_logger_handle(self):
#         touchfile(self.configs['logfilepath'])
#         self.logger_handle = Logger(self.configs['logfilepath'])

# import torch
# from torch.utils import data

# import copy
# import numpy as np
# import time
# import cv2
# import os
# from modules import (
#     Logger, touchdir, touchfile, BuildDataset, BuildChangeDetector, BuildOptimizer, BuildScheduler, BuildLoss, metrics)


# class BinaryChangeDetectionFramework():
#     def __init__(self, configs):
#         super().__init__()
#         self.configs = configs
#         self._get_device()
#         self._get_CD_model_using_name()
        
#         # ─────────────────────────────────────────────────────────────
#         # 1. 确定保存目录
#         self.snapshot_dir = os.path.join(self.configs["train"]["snapshots_dir"], self.configs["model_name"])
#         os.makedirs(self.snapshot_dir, exist_ok=True)

#         # 2. 智能获取数据集名称 (用于区分最佳模型)
#         self.dataset_name = self._get_dataset_name()
        
#         # 3. 定义记录文件和模型路径
#         # 记录文件: ./checkpoints/A2Net/best_record_levir.txt
#         # 模型文件: ./checkpoints/A2Net/best_iou_levir.pth
#         self.best_record_path = os.path.join(self.snapshot_dir, f'best_record_{self.dataset_name}.txt')
#         self.best_model_path = os.path.join(self.snapshot_dir, f'best_iou_{self.dataset_name}.pth')

#         # 4. 加载数据和日志
#         self._get_CD_dataloader()
#         self._get_logger_handle()
        
#         # 5. 初始化最佳 IoU (尝试从 txt 读取，防止重启训练后被重置)
#         self.best_iou = self._load_best_iou_record()
#         # ─────────────────────────────────────────────────────────────

#     def _get_dataset_name(self):
#         """尝试从配置中提取数据集名称"""
#         if 'dataset' in self.configs and 'name' in self.configs['dataset']:
#             return self.configs['dataset']['name']
        
#         if '_BASE_CONFIG_' in self.configs:
#             base_path = self.configs['_BASE_CONFIG_']
#             filename = os.path.basename(base_path) 
#             name = os.path.splitext(filename)[0]   
#             return name
            
#         return 'unknown_dataset'

#     def _load_best_iou_record(self):
#         """启动时读取历史最佳记录"""
#         if os.path.exists(self.best_record_path):
#             try:
#                 with open(self.best_record_path, 'r') as f:
#                     val = float(f.read().strip())
#                 print(f"Loaded existing best IoU record: {val:.4f} for {self.dataset_name}")
#                 return val
#             except:
#                 return 0.0
#         return 0.0

#     def training(self):
#         self.logger_handle.info(f'Starting Training on {self.dataset_name} | Previous Best IoU: {self.best_iou:.4f}')
#         self._get_optim()
#         self._get_lr_scheduler()
#         self._get_loss()
        
#         for epoch in range(self.configs["train"]["epochs"]):
#             loss_list=[]
#             self.CD_model.train()
#             for i,batch in enumerate(self.CD_dataloader_train):
#                 self.optimizer.zero_grad()
#                 pre_tensor, post_tensor, label_tensor, fname = batch["image1"], batch["image2"], batch["gt"], batch["fname"]
#                 pre_tensor = pre_tensor.to(self.device)
#                 post_tensor = post_tensor.to(self.device)
                
#                 if self.configs['model']['output_channels'] < 2:
#                     label_tensor = (label_tensor > 0).float().to(self.device) 
#                 else:
#                     label_tensor = label_tensor.type(torch.LongTensor).to(self.device)
                    
#                 prediction = self.CD_model(pre_tensor, post_tensor)
    
#                 if self.configs['model']['output_channels'] < 2:
#                     total_loss = self.loss(prediction,label_tensor)
#                 else:
#                     total_loss = self.loss(prediction,(label_tensor.squeeze(1)))
                
#                 loss_list.append(total_loss.item()) 
#                 total_loss.backward()
#                 self.optimizer.step()
            
#             torch.cuda.empty_cache() 
#             loss_avg = sum(loss_list)/len(loss_list)
#             self.scheduler.step(loss_avg) 
#             lr = self.optimizer.param_groups[0]['lr']
            
#             self.logger_handle.info(time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time())) + 
#                                     f', epoch={epoch+1} | loss={loss_avg:.7f} | lr={lr:.7f}')
            
#             # ──────────────── 修改点：只进行 Eval，不保存常规 epoch 模型 ────────────────
#             if (epoch+1) % self.configs["train"]["save_intervals"] == 0:
#                 self.current_epoch = epoch + 1
                
#                 # [已删除] self._save_checkpoint(is_best=False) 
#                 # 这里我们不再保存 1.pth, 2.pth 等文件
                
#                 if self.configs["eval"]:
#                     self.eval() # 进入 Eval，如果效果好，Eval 内部会调用保存最佳模型

#     def eval(self):
#         self.CD_model.eval()      
#         all_preds = []
#         all_gts = []
        
#         with torch.no_grad():
#             for _, batch in enumerate(self.CD_dataloader_eval):
#                 pre_tensor, post_tensor, label_tensor, fname = (
#                     batch["image1"].to(self.device),
#                     batch["image2"].to(self.device),
#                     batch["gt"].to(self.device),
#                     batch["fname"],
#                 )
                
#                 probs = self.CD_model(pre_tensor, post_tensor)

#                 if self.configs['model']['output_channels'] < 2:
#                     prediction = (probs > 0.5).to(torch.uint8)
#                     label_tensor = (label_tensor > 0).to(torch.uint8) 
#                 else:
#                     prediction = torch.argmax(probs, dim=1).to(torch.uint8)
#                     label_tensor = label_tensor.to(torch.uint8)

#                 all_preds.append(prediction.cpu().numpy())
#                 all_gts.append(label_tensor.cpu().numpy())

#         y_pred_flat = np.concatenate([p.ravel() for p in all_preds])
#         y_true_flat = np.concatenate([g.ravel() for g in all_gts])

#         # 计算指标
#         scores = self._calculate_metrics(y_pred_flat, y_true_flat)
        
#         log_msg = (f"Epoch {self.current_epoch} Eval ({self.dataset_name}): "
#                    f"IoU={scores['iou']:.4f}, F1={scores['f1']:.4f}, "
#                    f"Pre={scores['precision']:.4f}, Rec={scores['recall']:.4f}, "
#                    f"Acc={scores['accuracy']:.4f}")
#         self.logger_handle.info(log_msg)

#         # ──────────────── Best Model Logic ────────────────
#         # 只有当指标打破历史记录时，才保存
#         if scores['iou'] > self.best_iou:
#             previous_best = self.best_iou
#             self.best_iou = scores['iou']
            
#             self.logger_handle.info(f"!!! NEW BEST IoU: {self.best_iou:.4f} (Was {previous_best:.4f}) !!!")
            
#             # 1. 保存最佳模型 (覆盖旧的 best_iou_levir.pth)
#             self._save_best_checkpoint()
            
#             # 2. 更新记录文件
#             with open(self.best_record_path, 'w') as f:
#                 f.write(str(self.best_iou))

#         torch.cuda.empty_cache()    

#     def testing(self):
#         self.logger_handle.info(f"Testing on {self.dataset_name}...")
#         self.CD_model.eval()
#         touchdir(self.configs["test"]["save_path"])

#         all_preds, all_gts = [], []

#         with torch.no_grad():
#             for _, data in enumerate(self.CD_dataloader_test):
#                 pre_tensor  = data["image1"].to(self.device)
#                 post_tensor = data["image2"].to(self.device)
#                 label_tensor = data["gt"].to(self.device)

#                 logits = self.CD_model(pre_tensor, post_tensor)
#                 probs  = logits

#                 if self.configs['model']['output_channels'] < 2:
#                     prediction   = (probs > 0.5).to(torch.uint8)
#                     label_tensor = (label_tensor > 0).to(torch.uint8)
#                 else:
#                     prediction   = torch.argmax(probs, dim=1).to(torch.uint8)
#                     label_tensor = label_tensor.to(torch.uint8)

#                 result = prediction.cpu().numpy()
#                 if result.ndim == 4 and result.shape[1] == 1:
#                     result = result.squeeze(1)
                
#                 for j in range(result.shape[0]):
#                     mask = result[j]
#                     mask = (mask * 255).astype(np.uint8)
#                     save_name = os.path.join(self.configs["test"]["save_path"], data["fname"][j])
#                     cv2.imwrite(save_name, mask)

#                 all_preds.append(result)
#                 all_gts.append(label_tensor.cpu().numpy())

#         y_pred = np.concatenate([p.ravel() for p in all_preds]).astype(np.uint8)
#         y_true = np.concatenate([g.ravel() for g in all_gts]).astype(np.uint8)
        
#         scores = self._calculate_metrics(y_pred, y_true)
        
#         self.logger_handle.info("---------------- Test Results ----------------")
#         self.logger_handle.info(f"Dataset  : {self.dataset_name}")
#         self.logger_handle.info(f"IoU      : {scores['iou']:.4f}")
#         self.logger_handle.info(f"F1       : {scores['f1']:.4f}")
#         self.logger_handle.info(f"Precision: {scores['precision']:.4f}")
#         self.logger_handle.info(f"Recall   : {scores['recall']:.4f}")
#         self.logger_handle.info(f"Accuracy : {scores['accuracy']:.4f}")
#         self.logger_handle.info("----------------------------------------------")
        
#         torch.cuda.empty_cache()

#     def _calculate_metrics(self, pred, target):
#         pred = pred > 0
#         target = target > 0
        
#         tp = np.logical_and(pred, target).sum()
#         tn = np.logical_and(~pred, ~target).sum()
#         fp = np.logical_and(pred, ~target).sum()
#         fn = np.logical_and(~pred, target).sum()
        
#         eps = 1e-6
#         accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
#         precision = tp / (tp + fp + eps)
#         recall = tp / (tp + fn + eps)
#         f1 = 2 * (precision * recall) / (precision + recall + eps)
#         iou = tp / (tp + fp + fn + eps)
        
#         return {
#             "iou": iou, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy
#         }

#     def _get_CD_model_using_name(self):
#         self.changedecetor = BuildChangeDetector(changedetector_cfg=copy.deepcopy(self.configs['model']))        
#         self.CD_model = self.changedecetor.to(self.device)
#         if self.configs["mode"] == "test":
#             self.CD_model.load_state_dict(torch.load(self.configs["test"]["checkpoint"]))

#     def _get_CD_dataloader(self):
#         if self.configs["mode"]=="train":
#             CD_dataset_train = BuildDataset(mode='train',logger_handle=None,dataset_cfg=self.configs['dataset'])
#             self.CD_dataloader_train=data.DataLoader(dataset=CD_dataset_train,batch_size=self.configs["train"]["batch_size"],shuffle=True,num_workers=0,pin_memory=False)
#             if self.configs["eval"]:
#                 CD_dataset_eval = BuildDataset(mode='val',logger_handle=None,dataset_cfg=self.configs['dataset'])
#                 self.CD_dataloader_eval=data.DataLoader(dataset=CD_dataset_eval,batch_size=self.configs["eval"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False)
#         elif self.configs["mode"]=="test":
#             CD_dataset_test = BuildDataset(mode='test',logger_handle=None,dataset_cfg=self.configs['dataset'])
#             self.CD_dataloader_test=data.DataLoader(dataset=CD_dataset_test,batch_size=self.configs["test"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False)

#     def _get_optim(self):    
#         self.optimizer = BuildOptimizer(self.changedecetor, self.configs['scheduler']['optimizer'])

#     def _get_lr_scheduler(self):
#         self.scheduler = BuildScheduler(optimizer=self.optimizer, scheduler_cfg=self.configs['scheduler'])

#     def _get_loss(self):
#         self.loss = BuildLoss(loss_cfg=self.configs['losses']['loss_cls'])
        
#     def _get_device(self):
#         self.device = self.configs["device"]
    
#     def _save_best_checkpoint(self):
#         """仅保存最佳模型"""
#         # 保存最佳模型: ./checkpoints/A2Net/best_iou_levir.pth
#         save_path = self.best_model_path
#         torch.save(self.CD_model.state_dict(), save_path)
#         self.logger_handle.info(f"Saved Best Model to: {save_path}")

#     def _get_logger_handle(self):        
#         touchfile(self.configs['logfilepath'])
#         self.logger_handle = Logger(self.configs['logfilepath'])

import torch
from torch.utils import data

import copy
import numpy as np
import time
import cv2
import os
from modules import (
    Logger, touchdir, touchfile, BuildDataset, BuildChangeDetector, BuildOptimizer, BuildScheduler, BuildLoss, metrics)


class BinaryChangeDetectionFramework():
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self._get_device()
        self._get_CD_model_using_name()
        
        # ─────────────────────────────────────────────────────────────
        # 1. 确定保存目录
        self.snapshot_dir = os.path.join(self.configs["train"]["snapshots_dir"], self.configs["model_name"])
        os.makedirs(self.snapshot_dir, exist_ok=True)

        # 2. 智能获取数据集名称 (用于区分最佳模型)
        self.dataset_name = self._get_dataset_name()
        
        # 3. 定义记录文件和模型路径
        self.best_record_path = os.path.join(self.snapshot_dir, f'best_record_{self.dataset_name}.txt')
        self.best_model_path = os.path.join(self.snapshot_dir, f'best_iou_{self.dataset_name}.pth')

        # 4. 加载数据和日志
        self._get_CD_dataloader()
        self._get_logger_handle()
        
        # 5. 初始化最佳 IoU
        self.best_iou = self._load_best_iou_record()
        # ─────────────────────────────────────────────────────────────

    def _get_dataset_name(self):
        """尝试从配置中提取数据集名称"""
        if 'dataset' in self.configs and 'name' in self.configs['dataset']:
            return self.configs['dataset']['name']
        
        if '_BASE_CONFIG_' in self.configs:
            base_path = self.configs['_BASE_CONFIG_']
            filename = os.path.basename(base_path) 
            name = os.path.splitext(filename)[0]   
            return name
            
        return 'unknown_dataset'

    def _load_best_iou_record(self):
        """启动时读取历史最佳记录"""
        if os.path.exists(self.best_record_path):
            try:
                with open(self.best_record_path, 'r') as f:
                    val = float(f.read().strip())
                return val
            except:
                return 0.0
        return 0.0

    def training(self):
        self.logger_handle.info(f'Starting Training on {self.dataset_name} | Previous Best IoU: {self.best_iou:.4f}')
        self._get_optim()
        self._get_lr_scheduler()
        self._get_loss()
        
        for epoch in range(self.configs["train"]["epochs"]):
            loss_list=[]
            self.CD_model.train()
            for i,batch in enumerate(self.CD_dataloader_train):
                self.optimizer.zero_grad()
                pre_tensor, post_tensor, label_tensor, fname = batch["image1"], batch["image2"], batch["gt"], batch["fname"]
                pre_tensor = pre_tensor.to(self.device)
                post_tensor = post_tensor.to(self.device)
                
                if self.configs['model']['output_channels'] < 2:
                    label_tensor = (label_tensor > 0).float().to(self.device) 
                else:
                    label_tensor = label_tensor.type(torch.LongTensor).to(self.device)
                    
                prediction = self.CD_model(pre_tensor, post_tensor)
    
                if self.configs['model']['output_channels'] < 2:
                    total_loss = self.loss(prediction,label_tensor)
                else:
                    total_loss = self.loss(prediction,(label_tensor.squeeze(1)))
                
                loss_list.append(total_loss.item()) 
                total_loss.backward()
                self.optimizer.step()
            
            torch.cuda.empty_cache() 
            loss_avg = sum(loss_list)/len(loss_list)
            self.scheduler.step(loss_avg) 
            lr = self.optimizer.param_groups[0]['lr']
            
            self.logger_handle.info(time.strftime('%Y-%m-%d_%H_%M_%S',time.localtime(time.time())) + 
                                    f', epoch={epoch+1} | loss={loss_avg:.7f} | lr={lr:.7f}')
            
            if (epoch+1) % self.configs["train"]["save_intervals"] == 0:
                self.current_epoch = epoch + 1
                if self.configs["eval"]:
                    self.eval() 

    def eval(self):
        self.CD_model.eval()      
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for _, batch in enumerate(self.CD_dataloader_eval):
                pre_tensor, post_tensor, label_tensor, fname = (
                    batch["image1"].to(self.device),
                    batch["image2"].to(self.device),
                    batch["gt"].to(self.device),
                    batch["fname"],
                )
                
                probs = self.CD_model(pre_tensor, post_tensor)

                if self.configs['model']['output_channels'] < 2:
                    prediction = (probs > 0.5).to(torch.uint8)
                    label_tensor = (label_tensor > 0).to(torch.uint8) 
                else:
                    prediction = torch.argmax(probs, dim=1).to(torch.uint8)
                    label_tensor = label_tensor.to(torch.uint8)

                all_preds.append(prediction.cpu().numpy())
                all_gts.append(label_tensor.cpu().numpy())

        y_pred_flat = np.concatenate([p.ravel() for p in all_preds])
        y_true_flat = np.concatenate([g.ravel() for g in all_gts])

        # 计算指标
        scores = self._calculate_metrics(y_pred_flat, y_true_flat)
        
        log_msg = (f"Epoch {self.current_epoch} Eval ({self.dataset_name}): "
                   f"IoU={scores['iou']:.4f}, F1={scores['f1']:.4f}, "
                   f"Pre={scores['precision']:.4f}, Rec={scores['recall']:.4f}, "
                   f"Acc={scores['accuracy']:.4f}")
        self.logger_handle.info(log_msg)

        if scores['iou'] > self.best_iou:
            previous_best = self.best_iou
            self.best_iou = scores['iou']
            self.logger_handle.info(f"!!! NEW BEST IoU: {self.best_iou:.4f} (Was {previous_best:.4f}) !!!")
            self._save_best_checkpoint()
            with open(self.best_record_path, 'w') as f:
                f.write(str(self.best_iou))

        torch.cuda.empty_cache()    

    def testing(self):
        self.logger_handle.info(f"Testing on {self.dataset_name} with Color-Coded Visualization...")
        self.CD_model.eval()
        
        save_dir = self.configs["test"]["save_path"]
        touchdir(save_dir)

        all_preds, all_gts = [], []

        with torch.no_grad():
            for _, data in enumerate(self.CD_dataloader_test):
                pre_tensor  = data["image1"].to(self.device)
                post_tensor = data["image2"].to(self.device)
                label_tensor = data["gt"].to(self.device)

                # 推理
                logits = self.CD_model(pre_tensor, post_tensor)
                probs  = logits

                # 获取预测结果 (0 or 1)
                if self.configs['model']['output_channels'] < 2:
                    prediction   = (probs > 0.5).to(torch.uint8)
                    label_tensor = (label_tensor > 0).to(torch.uint8)
                else:
                    prediction   = torch.argmax(probs, dim=1).to(torch.uint8)
                    label_tensor = label_tensor.to(torch.uint8)

                # 转换 numpy
                pred_batch = prediction.cpu().numpy() # [B, 1, H, W] 或 [B, H, W]
                gt_batch   = label_tensor.cpu().numpy() # [B, 1, H, W] 或 [B, H, W]

                # 统一挤压维度到 [B, H, W]
                if pred_batch.ndim == 4 and pred_batch.shape[1] == 1:
                    pred_batch = pred_batch.squeeze(1)
                if gt_batch.ndim == 4 and gt_batch.shape[1] == 1:
                    gt_batch = gt_batch.squeeze(1)
                
                # ──────────────── 保存逻辑 (彩色可视化) ────────────────
                for j in range(pred_batch.shape[0]):
                    # 获取单张图的 预测掩膜 和 真值掩膜 (0/1)
                    p_mask = pred_batch[j] 
                    g_mask = gt_batch[j]
                    
                    # 初始化彩色画布 (H, W, 3) BGR格式
                    h, w = p_mask.shape
                    vis_img = np.zeros((h, w, 3), dtype=np.uint8)
                    
                    # 1. TP (True Positive): 预测=1, 真值=1 -> 白色
                    vis_img[(p_mask == 1) & (g_mask == 1)] = [255, 255, 255]
                    
                    # 2. TN (True Negative): 预测=0, 真值=0 -> 黑色 (默认就是0)
                    
                    # 3. FP (False Positive): 预测=1, 真值=0 -> 红色 (虚检)
                    # BGR: Red is [0, 0, 255]
                    vis_img[(p_mask == 1) & (g_mask == 0)] = [0, 0, 255]
                    
                    # 4. FN (False Negative): 预测=0, 真值=1 -> 蓝色 (漏检)
                    # BGR: Blue is [255, 0, 0]
                    vis_img[(p_mask == 0) & (g_mask == 1)] = [255, 0, 0]

                    # 命名逻辑
                    if "name1" in data and "name2" in data:
                        name1 = os.path.splitext(os.path.basename(data["name1"][j]))[0]
                        name2 = os.path.splitext(os.path.basename(data["name2"][j]))[0]
                        save_filename = f"{name1}_vs_{name2}_vis.png"
                    else:
                        raw_name = data["fname"][j]
                        base_name = os.path.splitext(os.path.basename(raw_name))[0]
                        save_filename = f"{base_name}_vis.png"

                    save_full_path = os.path.join(save_dir, save_filename)
                    cv2.imwrite(save_full_path, vis_img)

                all_preds.append(pred_batch)
                all_gts.append(gt_batch)

        # 计算并打印指标
        y_pred = np.concatenate([p.ravel() for p in all_preds]).astype(np.uint8)
        y_true = np.concatenate([g.ravel() for g in all_gts]).astype(np.uint8)
        
        scores = self._calculate_metrics(y_pred, y_true)
        
        self.logger_handle.info("---------------- Test Results ----------------")
        self.logger_handle.info(f"Dataset  : {self.dataset_name}")
        self.logger_handle.info(f"IoU      : {scores['iou']:.4f}")
        self.logger_handle.info(f"F1       : {scores['f1']:.4f}")
        self.logger_handle.info(f"Precision: {scores['precision']:.4f}")
        self.logger_handle.info(f"Recall   : {scores['recall']:.4f}")
        self.logger_handle.info(f"Accuracy : {scores['accuracy']:.4f}")
        self.logger_handle.info("----------------------------------------------")
        
        torch.cuda.empty_cache()

    def _calculate_metrics(self, pred, target):
        pred = pred > 0
        target = target > 0
        
        tp = np.logical_and(pred, target).sum()
        tn = np.logical_and(~pred, ~target).sum()
        fp = np.logical_and(pred, ~target).sum()
        fn = np.logical_and(~pred, target).sum()
        
        eps = 1e-6
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        iou = tp / (tp + fp + fn + eps)
        
        return {
            "iou": iou, "precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy
        }

    def _get_CD_model_using_name(self):
        self.changedecetor = BuildChangeDetector(changedetector_cfg=copy.deepcopy(self.configs['model']))        
        self.CD_model = self.changedecetor.to(self.device)
        if self.configs["mode"] == "test":
            self.CD_model.load_state_dict(torch.load(self.configs["test"]["checkpoint"]))

    def _get_CD_dataloader(self):
        if self.configs["mode"]=="train":
            CD_dataset_train = BuildDataset(mode='train',logger_handle=None,dataset_cfg=self.configs['dataset'])
            self.CD_dataloader_train=data.DataLoader(dataset=CD_dataset_train,batch_size=self.configs["train"]["batch_size"],shuffle=True,num_workers=0,pin_memory=False)
            if self.configs["eval"]:
                CD_dataset_eval = BuildDataset(mode='val',logger_handle=None,dataset_cfg=self.configs['dataset'])
                self.CD_dataloader_eval=data.DataLoader(dataset=CD_dataset_eval,batch_size=self.configs["eval"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False)
        elif self.configs["mode"]=="test":
            CD_dataset_test = BuildDataset(mode='test',logger_handle=None,dataset_cfg=self.configs['dataset'])
            self.CD_dataloader_test=data.DataLoader(dataset=CD_dataset_test,batch_size=self.configs["test"]["batch_size"],shuffle=False,num_workers=0,pin_memory=False)

    def _get_optim(self):    
        self.optimizer = BuildOptimizer(self.changedecetor, self.configs['scheduler']['optimizer'])

    def _get_lr_scheduler(self):
        self.scheduler = BuildScheduler(optimizer=self.optimizer, scheduler_cfg=self.configs['scheduler'])

    def _get_loss(self):
        self.loss = BuildLoss(loss_cfg=self.configs['losses']['loss_cls'])
        
    def _get_device(self):
        self.device = self.configs["device"]
    
    def _save_best_checkpoint(self):
        """仅保存最佳模型"""
        save_path = self.best_model_path
        torch.save(self.CD_model.state_dict(), save_path)
        self.logger_handle.info(f"Saved Best Model to: {save_path}")

    def _get_logger_handle(self):        
        touchfile(self.configs['logfilepath'])
        self.logger_handle = Logger(self.configs['logfilepath'])