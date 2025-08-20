# Unified Multi-Framework Model Zoo for Remote Sensing Change Detection  
<img width="5497" height="1013" alt="logo" src="logo.png" />

We are happy to announce that we are going to release a **unified model zoo** for **remote sensing change detection**.  

## ✨ Key Features  
- **Modular and Extensible**  
  The framework is designed with modular components, making it easy to extend and integrate new models or datasets.  

- **Lightweight, No Third-Party Toolbox Dependency**  
  Unlike existing solutions that rely on toolboxes such as *mmsegmentation*, our model zoo is implemented in a clean and lightweight way, with no heavy external dependencies.  

- **Multi-Framework Support**  
  Out-of-the-box support for multiple deep learning frameworks, including:  
  - PyTorch  
  - MindSpore  
  - LuoJiaNET  

## 🚀 Goals  
- Provide a unified and standardized platform for remote sensing change detection research.  
- Lower the entry barrier for new researchers and developers.  
- Facilitate fair comparisons and reproducibility across different models and frameworks.  


## 📌 Supported Models

The current release supports a wide range of classical and state-of-the-art change detection models:

| #  | Model        | Venue & Year     | Reference |
|----|-------------|-----------------|-----------|
| 1  | **FC-EF**        | IEEE ICIP 2018  | [Link](https://ieeexplore.ieee.org/document/8451652) |
| 2  | **FC-Siam-diff** | IEEE ICIP 2018  | [Link](https://ieeexplore.ieee.org/document/8451652) |
| 3  | **FC-Siam-conc** | IEEE ICIP 2018  | [Link](https://ieeexplore.ieee.org/document/8451652) |
| 4  | **FCNPP**        | IEEE GRSL 2019  | [Link](https://www.mdpi.com/2076-3417/9/9/1816) |
| 5  | **DSIFN**        | ISPRS 2020      | [Link](https://www.sciencedirect.com/science/article/abs/pii/S0924271620301532) |
| 6  | **FDCNN**        | IEEE TGRS 2020  | [Link](https://ieeexplore.ieee.org/document/9052762) |
| 7  | **SNUNet**       | IEEE GRSL 2021  | [Link](https://ieeexplore.ieee.org/abstract/document/9355573) |
| 8  | **BIT**          | IEEE TGRS 2021  | [Link](https://ieeexplore.ieee.org/document/9491802) |
| 9  | **DSAMNet**      | IEEE TGRS 2021  | [Link](https://ieeexplore.ieee.org/document/9467555) |
| 10 | **ChangeFormer** | IEEE IGARSS 2022| [Link](https://ieeexplore.ieee.org/document/9883686) |
| 11 | **RDP-Net**      | IEEE TGRS 2022  | [Link](https://ieeexplore.ieee.org/document/9970750) |
| 12 | **ICIFNet**      | IEEE TGRS 2022  | [Link](https://ieeexplore.ieee.org/abstract/document/9759285) |
| 13 | **HANet**        | IEEE JSTARS 2023| [Link](https://ieeexplore.ieee.org/abstract/document/10093022) |
| 14 | **TinyCD**       | Neural Computing and Applications 2023 | [Link](https://link.springer.com/article/10.1007/s00521-022-08122-3) |
| 15 | **USSFC-Net**    | IEEE TGRS 2023  | [Link](https://ieeexplore.ieee.org/document/10081023) |
| 16 | **A2Net**        | IEEE TGRS 2023  | [Link](https://ieeexplore.ieee.org/abstract/document/10034814) |
| 17 | **DMINet**       | IEEE TGRS 2023  | [Link](https://ieeexplore.ieee.org/document/10034787) |

---

## 📂 Supported Datasets

Our framework supports a wide range of public change detection benchmarks:

| #  | Dataset        | Link |
|----|----------------|------|
| 1  | **WHU Building** | [Link](http://gpcv.whu.edu.cn/data/) |
| 2  | **LEVIR-CD**     | [Link](https://justchenhao.github.io/LEVIR/) |
| 3  | **GVLM**         | [Link](https://github.com/zxk688/GVLM) |
| 4  | **CropLand Change Detection (CLCD)** | [Link](https://github.com/liumency/CropLand-CD) |
| 5  | **SYSU-CD**      | [Link](https://github.com/liumency/SYSU-CD) |
| 6  | **EGY-BCD**      | [Link](https://github.com/oshholail/EGY-BCD) |
| 7  | **HRCUS-CD**     | [Link](https://github.com/zjd1836/AERNet#hrcus-cd) |
| 8  | **OSCD (Onera Satellite Change Detection)** | [Link](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection#files) |
| 9  | **SI-BU**        | [Link](https://github.com/liaochengcsu/BCE-Net) |
| 10 | **DSIFN Dataset**| [Link](https://github.com/GeoZcx/A-deeply-supervised-image-fusion-network-for-change-detection-in-remote-sensing-images/tree/master/dataset) |
| 11 | **Google Dataset** | [Link](https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery) |
| 12 | **CNAM-CD**      | [Link](https://github.com/Silvestezhou/CNAM-CD) |
| 13 | **S2Looking**    | [Link](https://github.com/S2Looking/Dataset) |
| 14 | **HRSCD**        | [Link](https://ieee-dataport.org/open-access/hrscd-high-resolution-semantic-change-detection-dataset) |
| 15 | **SECOND**       | [Link](https://drive.google.com/u/0/uc?id=1mN8jzCKKK27p3ODGoDgepjiRYGQpB34u&export=download) |
| 16 | **Hi-UCD**       | [Link](https://github.com/Daisy-7/Hi-UCD-S) |
| 17 | **WUSU**         | [Link](https://github.com/angienikki/openwusu) |
| 18 | **MSBC & MSOSCD** | [Link](https://github.com/Lihy256/MSCDUnet) |
| 19 | **xBD (xView2)** | [Link](https://xview2.org/) |

---





