<div align="left" markdown>
<div align="center">
  <img src="resources/ChangeDetection.png" width="600"/>
  <div>&nbsp;</div>
  <div align="center">
    <b><font size="5">LuoJiaNet官网</font></b>
    <sup>
      <a href="http://58.48.42.237/luojiaNet/home">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">LuoJiaSet官网</font></b>
    <sup>
      <a href="http://58.48.42.237/luojiaSet/home">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>
  </div>

<div align="center">
     <a href="https://github.com/WHULuoJiaTeam/luojianet">
        <img src="resources/svg/python.svg" width="210" />
      </a>
     <a href="https://github.com/WHULuoJiaTeam/luojianet">
         <img src="resources/svg/build.svg" width="130" />
      </a>
     <a href="https://github.com/WHULuoJiaTeam/luojianet/issues">
         <img src="resources/svg/issue.svg" width="150" />
      </a>
     <a href="https://github.com/WHULuoJiaTeam/luojianet/actions">
         <img src="resources/svg/license.svg" width="150" />
      </a>
</div>

<div align="center">
     <a href="http://58.48.42.237/luojiaNet/tutorial/quickstart">
        <i><font size="3">📝使用文档</font></i>
      </a>
     <i><font size="4"> | </font></i>
     <a href="https://github.com/WHULuoJiaTeam/luojianet">
        <i><font size="3">🚀git仓库</font></i>
      </a>
     <i><font size="4"> | </font></i>
    <a href="https://github.com/WHULuoJiaTeam/Model_Zoo">
        <i><font size="3">🎁模型仓库</font></i>
      </a>
</div>

<div align="center">
     <a href="README.md">
        <i><font size="3">English</font></i>
      </a>
     <i><font size="3"> | 中文</font></i>
</div>

## 简介

LuoJiaChangeDetection，简称luoJiaCD，是一个基于 [luojianet](http://58.48.42.237/luojiaNet/home) 开发的，致力于遥感变化检测相关技术研发的开源工具箱。它提供大量的遥感变化检测模型以及它们的预训练权重和训练策略。通过解耦的模块设计，您可以轻松地将luojiaCD应用到您自己的任务中。

主分支代码目前支持 **luojianet 1.8+** 以上的版本，包含 **luojianet 2.0🔥** 版本。

### 主要特性

- **高性能** luojiaCD集成了大量基于CNN和Transformer的高性能模型, 如DMINet、Changeformer，帮助用户快速选型并将其应用于变化检测任务。

- **灵活高效** luojiaCD基于高效的深度学习框架luojianet开发，具有自动并行和自动微分等特性，支持不同硬件平台上（CPU/GPU/Ascend），同时支持效率优化的静态图模式和调试灵活的动态图模式。

## 安装

```
cd LuoJiaChangeDetection
# 安装相关依赖
pip install -r requirements.txt
```



## 快速入门

### 上手教程
我们提供了系列教程，帮助用户学习如何使用luojiaCD，参见[教程](https://gitee.com/xiaokang-zhang/LuoJiaChangeDetection/blob/master/Demo.ipynb)。

### 模型训练

通过运行`main.py`，用户可以通过更改.yml文件或.txt中mode参数在标准数据集或自定义数据集上训练模型。
 #### 选择模型进行训练
    python main.py --config configs/daminet/daminet_levir_train.yml

训练权重文件将保存在`.yaml`文件参数snapshots_dir指定路径下。

### 模型推理和验证

通过运行`main.py`，用户可以通过更改.yml文件或.txt中mode参数在标准数据集或自定义数据集上测试。
 #### 选择模型进行测试
    python main.py --config configs/daminet/daminet_levir_test.yml


## 模型列表

* [A2Net](https://ieeexplore.ieee.org/abstract/document/10034814)
* [BISRNet](https://ieeexplore.ieee.org/document/9721305)
* [ChangeFormer](https://ieeexplore.ieee.org/document/9883686)
* [DMINet](https://ieeexplore.ieee.org/document/10034787)
* [DSIFN](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532)
* [FC-EF](https://ieeexplore.ieee.org/abstract/document/8451652)
* [FC-Siam](https://ieeexplore.ieee.org/abstract/document/8451652)
* [FC-Conc](https://ieeexplore.ieee.org/abstract/document/8451652)
* [FDCNN](https://ieeexplore.ieee.org/document/9052762)
* [HANet](https://ieeexplore.ieee.org/abstract/document/10093022)
* [RDPNet](https://ieeexplore.ieee.org/document/9970750)
* [SNUNet](https://ieeexplore.ieee.org/document/9355573)
* [TinyCD](https://link.springer.com/article/10.1007/s00521-022-08122-3)
* [USSFCNet](https://ieeexplore.ieee.org/document/10081023)


## 支持数据集

* [WHU Building](http://gpcv.whu.edu.cn/data/)		
* [LEVIR-CD](https://justchenhao.github.io/LEVIR/)		
* [GVLM](https://github.com/zxk688/GVLM)	
* [CropLand Change Dection (CLCD) Dataset](https://github.com/liumency/CropLand-CD)		
* [SYSU-CD](https://github.com/liumency/SYSU-CD)		
* [EGY-BCD Dataset](https://github.com/oshholail/EGY-BCD)		
* [HRCUS-CD](https://github.com/zjd1836/AERNet#hrcus-cd)	
* [OSCD](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection#files)		
* [SI-BU dataset](https://github.com/liaochengcsu/BCE-Net)		
* [DSIFN](https://github.com/GeoZcx/*A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset)		
* [Google](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)		
* [CNAM-CD](https://github.com/Silvestezhou/CNAM-CD)		
* [S2Looking](https://github.com/S2Looking/Dataset)		
* [HRSCD](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset)		
* [SECOND](https://drive.google.com/u/0/uc?id=1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u&export=download)		
* [Hi-UCD](https://github.com/Daisy-7/Hi-UCD-S)		
* [WUSU](https://github.com/angienikki/openwusu)		
* [Multisource built-up change (MSBC) and multisource OSCD (MSOSCD) datasets](https://github.com/Lihy256/MSCDUnet)		
* [xBD](https://xview2.org/)		



## 更新

12/28/2023 - luojiaCD上线！

## 贡献指南

欢迎开发者用户提issue或提交代码PR，或贡献更多的算法和模型，一起让luojiaCD变得更好。

有关贡献指南，请参阅 [CONTRIBUTING.md](CONTRIBUTING.md)。 请遵循模型编写指南所规定的规则来贡献模型接口。

## 许可证

本项目遵循Apache License 2.0开源协议。

## 致谢

 衷心感谢所有参与的研究人员和开发人员为这个项目所付出的努力。 

## 引用

如果你觉得luojiaCD对你的项目有帮助，请考虑引用。

