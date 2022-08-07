# Self-Supervised Feature Learning from Partial Point Clouds via Pose Disentanglement
PyTorch implementaton of our IROS 2022 paper "Self-Supervised Feature Learning from Partial Point Clouds via Pose Disentanglement".
You can visit our project website [here](https://ms-tsai.github.io/Partial-Point-Clouds-Disentangler-Project-Page/).

In this paper, we propose a novel self-supervised framework to learn informative features from partial point clouds. We leverage partial point clouds scanned by LiDAR that contain both content and pose attributes, and we show that disentangling such two factors from partial point clouds enhances feature learning.

<div align=center><img height="300" src="https://github.com/MS-Tsai/Partial-Point-Clouds-Disentangler/blob/main/sample/Teaser.png"/></div>

## Paper
[Self-Supervised Feature Learning from Partial Point Clouds via Pose Disentanglement](https://arxiv.org/abs/2201.03018)  
[Meng-Shiun Tsai*](mailto:infinitesky.cs08g@nctu.edu.tw), [Pei-Ze Chiang*](mailto:ztex080104518.cs08g@nctu.edu.tw), [Yi-Hsuan Tsai](https://sites.google.com/site/yihsuantsai/), [Wei-Chen Chiu](https://walonchiu.github.io/)  
IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022.

Please cite our paper if you find it useful for your research.  
```
@article{tsai2022self,
  title={Self-Supervised Feature Learning from Partial Point Clouds via Pose Disentanglement},
  author={Tsai, Meng-Shiun and Chiang, Pei-Ze and Tsai, Yi-Hsuan and Chiu, Wei-Chen},
  journal={arXiv preprint arXiv:2201.03018},
  year={2022}
}
```

## Environment Setting
* This code was developed with Python 3.7.9 & Pytorch 1.2.0 & CUDA 10.0

## Dataset
* For Complete/Partial ShapeNet datasets, you can download from [here](https://drive.google.com/drive/folders/1SDTE0sLYW5hwXRGJjQssBJnpn56fQW3J?usp=sharing)

## Compile our Extension Modules:
```
cd emd
python3 setup.py install
cd expansion_penalty
python3 setup.py install
```

## Pre-train Weight
* For pre-train weight of our pretext model, you can download from [here](https://drive.google.com/drive/folders/1KWmSd2WS8JU25vXOQkzYx0lY2Lza2H39?usp=sharing)

## Training
```
python train.py --gpu 0
```

## Acknowledgments
Our code is based on [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), and [Dynamic Graph CNN for Learning on Point Clouds](https://github.com/WangYueFt/dgcnn).  
The implementation of completion branch is borrowed from [MSN: Morphing and Sampling Network for Dense Point Cloud Completion](https://github.com/Colin97/MSN-Point-Cloud-Completion).
