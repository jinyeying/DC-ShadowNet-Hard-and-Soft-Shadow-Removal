# DC-ShadowNet (ICCV'2021)

## Introduction
This is an implementation of the following paper.
> [DC-ShadowNet: Single-Image Hard and Soft Shadow Removal Using
Unsupervised Domain-Classifier Guided Network.](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_DC-ShadowNet_Single-Image_Hard_and_Soft_Shadow_Removal_Using_Unsupervised_Domain-Classifier_ICCV_2021_paper.pdf)
>  International Conference on Computer Vision (ICCV'2021)

Yeying Jin, [Aashish Sharma](https://aasharma90.github.io/) and [Robby T. Tan](https://tanrobby.github.io/pub.html)

[[Paper Link]](https://openaccess.thecvf.com/content/ICCV2021/papers/Jin_DC-ShadowNet_Single-Image_Hard_and_Soft_Shadow_Removal_Using_Unsupervised_Domain-Classifier_ICCV_2021_paper.pdf) [[Supplement]](https://openaccess.thecvf.com/content/ICCV2021/supplemental/Jin_DC-ShadowNet_Single-Image_Hard_ICCV_2021_supplemental.pdf) [[Project Page]](https://github.com/jinyeying/DC-ShadowNet-Hard-and-Soft-Shadow-Removal) [[Slides]](https://www.dropbox.com/s/axgryynoqwdlepx/DC-ShadowNet_release.pdf?dl=0) [[Video]](https://www.dropbox.com/s/mbe82k4ckokm04o/03793.mp4?dl=0)

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


### Datasets
1. SRD (please download [train](https://drive.google.com/file/d/1W8vBRJYDG9imMgr9I2XaA13tlFIEHOjS/view) [BaiduNetdisk](https://pan.baidu.com/s/1mj3BoRQ) and [test from the authors](http://www.shengfenghe.com/publications/)).
[Extracted Shadow Masks in the SRD Dataset](https://github.com/vinthony/ghost-free-shadow-removal)

2. [AISTD](https://www3.cs.stonybrook.edu/~cvl/projects/SID/index.html) 

3. [LRSS: Soft Shadow Dataset](http://visual.cs.ucl.ac.uk/pubs/softshadows/)

4. [ISTD](https://github.com/DeepInsight-PCALab/ST-CGAN) 

5. [USR: Unpaired Shadow Removal Dataset](https://drive.google.com/file/d/1PPAX0W4eyfn1cUrb2aBefnbrmhB1htoJ/view)

### Shadow Removal Results:
<img src="teaser/hard_shadow.PNG" > 

1. SDR Dataset
[DC-ShadowNet-SDR](https://www.dropbox.com/s/278h3k1hyufvo0l/DC-ShadowNet_SRD.zip?dl=0),
[SDR-Results](https://www.dropbox.com/sh/kg87bt5tcmi535n/AACrGNvLgpWd-UTs6NWep9MLa?dl=0)

2. AISTD/ISTD+ Dataset
[DC-ShadowNet-AISTD](https://www.dropbox.com/s/uacmjnnmzfldh10/DC-ShadowNet_AISTD.zip?dl=0),
[AISTD-Results](https://www.dropbox.com/sh/foqmi8olum6n3qz/AADX3aQ4yzWvKHh4wtAF6YREa?dl=0)

3. LRSS Soft Shadow Dataset
[DC-ShadowNet-LRSS](https://www.dropbox.com/sh/i9rto8h1shbc315/AADa6kvxwtUP8EKju2jSKxn2a?dl=0),
[LRSS-Results](https://www.dropbox.com/sh/ryku9yr1j4u4898/AABC2gPoM9scASHZ0N6SmwBDa?dl=0)

<img src="teaser/soft_shadow.PNG" > 

4. ISTD Dataset
[DC-ShadowNet-ISTD](https://www.dropbox.com/s/vlo9ng6hufyxuxl/DC-ShadowNet_ISTD.zip?dl=0)
[ISTD-Results](https://www.dropbox.com/sh/gsipgdeb8w7ms04/AAC6C5xvtH4pp4WkHQBr8NeAa?dl=0)

5. USR Dataset
[DC-ShadowNet-USR](https://www.dropbox.com/s/ybmwxtmo7cdljyz/DC-ShadowNet_USR.zip?dl=0)
[USR-Results](https://www.dropbox.com/sh/0g63jv4fztpp64o/AABftwGguAERWeARRCk6Gybga?dl=0)

### Evaluation
The default root mean squared error (RMSE) evaluation code used by all methods (including ours) actually computes mean absolute error (MAE). 

1. The faster version [MAE evaluation code](https://www.dropbox.com/sh/nva9ddquvgogb5n/AABOHrWx9whMXeItcZfODe9ia?dl=0)
2. The original version [MAE evaluation code](https://drive.google.com/file/d/1-lG8nAJbWajAC4xopx7hGPKbuwYRw4x-/view)

1.1 SRD Dataset, set the paths of the shadow removal result and the dataset in demo_srd_release.m and then run it.

Get the following Table 1 in the main paper on the SRD dataset (size: 256x256).

| Method | Training | All | Shadow | Non-Shadow |
|------------------|----------|----------|------|------|
| **DC-ShadowNet** | Unpaired | **4.66** | 7.70 | 3.39 |
| Mask-ShadowGAN | Unpaired | 6.40 | 11.46 | 4.29 |
| DSC | Paired | 4.86 | 8.81 | **3.23** |
| DeShadowNet | Paired | 5.11 | **3.57** | 8.82 |
| Gong | Prior | 12.35 | 25.43 | 6.91 |
| Input Image | N/A | 13.77 | 37.40 | 3.96 |

1.2 AISTD Dataset, set the paths of the shadow removal result and the dataset in demo_aistd_release.m and then run it.

Get the following Table 2 in the main paper on the AISTD dataset (size: 256x256).
| Method | Training | All | Shadow | Non-Shadow |
|------------------|----------|---------|----------|-----|
| **DC-ShadowNet** | Unpaired | **4.6** | **10.3** | 3.5 |

1.3 LRSS Soft Shadow Dataset, set the paths of the shadow removal result and the dataset in demo_lrss_release.m and then run it.

Get the following Table 3 in the main paper on the LRSS dataset (size: 256x256).
| Method | Training | All | 
|------------------|----------|----------|
| **DC-ShadowNet** | Unpaired | **3.48** |
| Input Image | N/A | 12.26 |

## Pre-trained Model
1. Download the [pre-trained SRD model](https://www.dropbox.com/s/y7247t2pbozhlo7/SRD_params_0500000.pt?dl=0), put in results/SRD/model/

2. Download the [pre-trained AISTD model](https://www.dropbox.com/s/nv7i4ofal9s9wud/AISTD_params_0500000.pt?dl=0), put in results/AISTD/model/ 

3. Download the [pre-trained ISTD model](https://www.dropbox.com/s/0rro00nqg7tbds3/ISTD_params_0600000.pt?dl=0), put in results/ISTD/model/

### Test
python main_test.py --dataset SRD --datasetpath [path_to_SRD dataset] --use_original_name False

rename to the original name, please change the suffix of test images accordingly (.jpg OR .png)

python main_test.py --dataset SRD --datasetpath [path_to_SRD dataset] --use_original_name True --im_suf_A .jpg

Results: results/SRD/[iteration]/outputB; results/SRD/[iteration]/inputA_outputB

<img src="results/SRD/500000/inputA_outputB/IMG_6456.png" >

### Train
1. Implement the papers [On the removal of shadows from images (TPAMI,05)](https://www.cs.sfu.ca/~mark/ftp/Pami06/pami06.pdf) and [Recovery of Chromaticity Image Free from Shadows via Illumination Invariance (ICCV,03)](https://www.cs.sfu.ca/~mark/ftp/Iccv03ColorWkshp/iccv03wkshp.pdf)
<img src="teaser/chromaticity.png" > 

### Directory
2. Download Datasets and run 1. get the Shadow-Free Chromaticity Maps after Illumination Compensation, and put them in the trainC folder, you should see the following directory structure. 
```
${DC-ShadowNet-Hard-and-Soft-Shadow-Removal}
|-- dataset
    |-- SRD
      |-- trainA ## Shadow 
      |-- trainB ## Shadow-free 
      |-- trainC ## Shadow-Free Chromaticity Maps after Illumination Compensation
      |-- testA  ## Shadow 
      |-- testB  ## Shadow-free 
...
```

2. python main_train.py --dataset SRD --datasetpath [path_to_SRD dataset] --iteration [iteration]


## Shadow-Robust Feature
Get the following Figure 5 in the main paper, VGG feature visualization code is in feature_release folder,

python test_VGGfeatures.py

<img src="teaser/feature_map.png" > 

Results: ./results_VGGfeatures/shadow_VGGfeatures/layernumber/imagenumber/visual_featurenumber_RMSE.jpg


### Citation
If this work is useful for your research, please cite our paper. 
```
@inproceedings{jin2021dc,
  title={DC-ShadowNet: Single-Image Hard and Soft Shadow Removal Using Unsupervised Domain-Classifier Guided Network},
  author={Jin, Yeying and Sharma, Aashish and Tan, Robby T},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5027--5036},
  year={2021}
}
```
