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
<p align="center"><img src="teaser/network.png" width="98%"></p>

### Datasets
SRD Dataset (please download [train](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) and [test](http://www.cs.cityu.edu.hk/~rynson/papers/demos/cvpr17a_testset.rar) from the [authors](http://www.shengfenghe.com/publications/)).

[LRSS: Soft Shadow dataset](http://visual.cs.ucl.ac.uk/pubs/softshadows/)

[ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN) 

[AISTD](https://www3.cs.stonybrook.edu/~cvl/projects/SID/index.html) 

[USR: Unpaired Shadow Removal dataset](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)

### Shadow removal results:
1.The results of this paper on the SDR dataset can be downloaded here:

2.The results of this paper on the ISTD dataset can be downloaded here:

## Usage 
### Evaluation
The default root mean squared error (RMSE) evaluation code used by all methods (including ours) actually computes mean absolute error (MAE). 

The original [MAE evaluation code](https://drive.google.com/file/d/1-lG8nAJbWajAC4xopx7hGPKbuwYRw4x-/view)

The faster version

## Citation
Please kindly cite our paper if you are using our codes:
