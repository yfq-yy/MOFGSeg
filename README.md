#  Semi-Supervised Learning for Multi-Object Segmentation with  Fine-Grained Classes



Multi-object semantic segmentation has made significant progress. However, Few work has focused on multi-object semantic segmentation with fine-grained classes. There are two reasons. Firstly, as segmentation becomes more fine-grained, the annotation becomes more complex and requires extensive domain-specific expertise and substantial time resources. Secondly, fine-grained segmentation poses unique challenges due to high inter-class similarity and large intra-class variance. To address these problems, we propose a semi-supervised learning method by weighted constrastive learing and boundary-aware loss for fine-grained segmentation. Specifically, we employ student-teacher dual networks and combined with consistency regularization to generate pseudo-labeling for unlabeled images. We further present a weighted contrastive learning to increase the distribution distance between different classes in the feature space, and a boundary-aware module to enhance boundary cues features. Our method demonstrates superior mean Intersection over Union (MIoU) performance compared to existing semi-supervised semantic segmentation techniques across multiple datasets, including the fine-grained ocean dataset UAV-SEG, public fine-grained datasets TAS500, the remote sensing dataset LoveDA, and the general semantic segmentation dataset Pascal VOC.


## 🛠️ Usage

### 1. Environment

First, clone this repo:

```shell
git clone https://github.com/yfq-yy/MOFGSeg.git
cd MOFGSeg/
```

Then, create a new environment and install the requirements:
```shell
conda create -n mofgseg python=3.7
conda activate mofgseg
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install tensorboard
pip install six
pip install pyyaml
pip install -U openmim
mim install mmcv==1.6.2
pip install einops
pip install timm
```

### 2. Data Preparation & Pre-trained Weights

#### 2.1 Pascal VOC 2012 Dataset
Download the dataset with wget:
```shell
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/EcgD_nffqThPvSVXQz6-8T0B3K9BeUiJLkY_J-NvGscBVA\?e\=2b0MdI\&download\=1 -O pascal.zip
unzip pascal.zip
```
Then your file structure will be like:

```
├── VOC2012
    ├── JPEGImages
    └── SegmentationClass
    
```

Next, download the following [pretrained weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hwanggr_connect_ust_hk/Eobv9tk6a6RJqGXEDm2D_TcB2mEn4r2-BLDkotZHkd2l6w?e=fJBy7v).
```
├── ./pretrained_weights
    ├── mit_b2.pth
    ├── mit_b3.pth
    ├── mit_b4.pth
    └── mit_b5.pth
```

For example, mit-B5:
```shell
mkdir pretrained_weights
wget https://hkustconnect-my.sharepoint.com/:u:/g/personal/hwanggr_connect_ust_hk/ET0iubvDmcBGnE43-nPQopMBw9oVLsrynjISyFeGwqXQpw?e=9wXgso\&download\=1 -O ./pretrained_weights/mit_b5.pth
```


### 3. Training & Evaluating

```bash
# use torch.distributed.launch
sh scripts/train.sh <num_gpu> <port>
# to fully reproduce our results, the <num_gpu> should be set as 4 on all three datasets
# otherwise, you need to adjust the learning rate accordingly


To train on other datasets or splits, please modify
``dataset`` and ``split`` in  train.sh.

```

### 4. Results

Model weights and training logs will be released soon.

#### 4.1 PASCAL VOC 2012 _original_

| Splits | 1/16 | 1/8  | 1/4 | 1/2 | Full |
| :- | - | - | - | - | - |
| Weights of _**Ours**_ | 75.87 | 78.97 | 80.03 | 82.24 | 83.22 |

#### 4.2 PASCAL VOC 2012 _augmented_

| Splits | 1/16 | 1/8  | 1/4 | 1/2 |
| :- | - | - | - | - |
| Weights of _**Ours**_ | 78.67 | 79.36 | 79.79 | 80.98 |

#### 4.3 UAV-SEG

| Splits | 1/16 | 1/8  | 1/4 | 1/2 |
| :- | - | - | - | - |
| Weights of _**Ours**_ | 51.99 | 55.95 | 64.88 | 67.64 |

#### 4.4 Tas500

| Splits | 1/16 | 1/8  | 1/4 | 1/2 |
| :- | - | - | - | - |
| Weights of _**Ours**_ | 46.70 | 47.46 | 54.47 | 54.94 |

#### 4.5 loveda

| Splits | 1/16 | 1/8  | 1/4 | 1/2 |
| :- | - | - | - | - |
| Weights of _**Ours**_ | 55.95 | 55.48 | 56.20 | 56.79 |



## Acknowlegement
_**Ours**_ is built upon [AllSpark](https://github.com/xmed-lab/AllSpark.git), [UniMatch](https://github.com/LiheYoung/UniMatch) and [SegFormer](https://github.com/NVlabs/SegFormer). We thank their authors for making the source code publicly available.


