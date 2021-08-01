# DC-ShadowNet: Hard-and-Soft-Shadow-Removal

## Introduction
This is an implementation of the following paper.
**DC-ShadowNet: Single-Image Hard and Soft Shadow Removal Using
Unsupervised Domain-Classifier Guided Network. (ICCV'2021)** 
Yeying Jin, [Aashish Sharma](https://aasharma90.github.io/) and [Robby T. Tan](https://tanrobby.github.io/pub.html)

### Abstract
Shadow removal from a single image is generally still an open problem.
Most existing learning-based methods use supervised learning and require a large number of paired images (shadow and corresponding non-shadow images) for training.
A recent unsupervised method, Mask-ShadowGAN, addresses this limitation. 
However, it requires a binary mask to represent shadow regions, making it inapplicable to soft shadows. 
To address the problem, in this paper, we propose an unsupervised domain-classifier guided shadow removal network, DC-ShadowNet. 
Specifically, we propose to integrate a shadow/shadow-free domain classifier into a generator and its discriminator, enabling them to focus on shadow regions.
To train our network, we introduce novel losses based on physics-based shadow-free chromaticity, shadow-robust perceptual features, and boundary smoothness. 
Moreover, we show that our unsupervised network can be used for test-time training that further improves the results. 
Our experiments show that all these novel components allow our method to handle soft shadows, and also to perform better on hard shadows both quantitatively and qualitatively than the existing state-of-the-art shadow removal methods.



Overview of the proposed method:
<p align="center"><img src="network.png" width="98%"></p>

### Datasets
SRD Dataset (please download [train](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) and [test](http://www.cs.cityu.edu.hk/~rynson/papers/demos/cvpr17a_testset.rar) from the [authors](http://www.shengfenghe.com/publications/)).

[LRSS: Soft Shadow dataset](http://visual.cs.ucl.ac.uk/pubs/softshadows/)

[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN) 

[AISTD](https://www3.cs.stonybrook.edu/~cvl/projects/SID/index.html) 

[USR: Unpaired Shadow Removal dataset](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)

### Shadow removal results:
1.[DC-ShadowNet](https://www.dropbox.com/sh/jhm4kxvq9apubq9/AAB5BickFfGhunK5ezJK0R0_a?dl=0) results on the SDR dataset,
[All results](https://www.dropbox.com/sh/kg87bt5tcmi535n/AACrGNvLgpWd-UTs6NWep9MLa?dl=0)

2.The results of this paper on the ISTD:

## Usage 
### Evaluation
The default root mean squared error (RMSE) evaluation code used by all methods (including ours) actually computes mean absolute error (MAE). 

1.The faster version [MAE evaluation code](https://www.dropbox.com/sh/nva9ddquvgogb5n/AABOHrWx9whMXeItcZfODe9ia?dl=0)
Set the paths of the shadow removal result and the dataset in demo_srd_release.m and then run it.
Get the following Table 1 in the main paper.

2.The original [MAE evaluation code](https://drive.google.com/file/d/1-lG8nAJbWajAC4xopx7hGPKbuwYRw4x-/view)

## Citation
Please kindly cite our paper if you are using our codes:
