# Visual Prompting for Generalized Few-shot Segmentation: A Multi-scale Approach (CVPR 2024)
*Mir Rayat Imtiaz Hossain, Mennatullah Siam, Leonid Sigal, James J. Little*
 
This repository contains source code for our **CVPR 2024** paper titled, [Visual Prompting for Generalized Few-shot Segmentation: A Multi-scale Approach](https://openaccess.thecvf.com/content/CVPR2024/papers/Hossain_Visual_Prompting_for_Generalized_Few-shot_Segmentation_A_Multi-scale_Approach_CVPR_2024_paper.pdf).

## &#x1F3AC; Getting Started

### :one: Requirements
We used `Python 3.9.0` in our experiments and the list of packages is available in the `requirements.txt` file. You can install them using `pip install -r requirements.txt`.

#### Setting up CUDA kernel for MSDeformAttn

After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:
```
cd VisualPromptGFSS/src/model/ops/
sh make.sh
```


### :two: Dataset

We used the versions of PASCAL and MS-COCO provided by [DIaM](https://github.com/sinahmr/DIaM). You can download the dataset from [here](https://etsmtl365-my.sharepoint.com/personal/seyed-mohammadsina_hajimiri_1_ens_etsmtl_ca/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fseyed%2Dmohammadsina%5Fhajimiri%5F1%5Fens%5Fetsmtl%5Fca%2FDocuments%2FDIaM%2Fdatasets%2Ezip&parent=%2Fpersonal%2Fseyed%2Dmohammadsina%5Fhajimiri%5F1%5Fens%5Fetsmtl%5Fca%2FDocuments%2FDIaM&ga=1).

The data folder should look like this:

```
data
├── coco
│   ├── annotations
│   ├── train
│   ├── train2014
│   ├── val
│   └── val2014
└── pascal
|   ├── JPEGImages
|   └── SegmentationClassAug
```

#### The train/val split

The train/val split can be found in the diectory `src/lists/`. We borrowed the list from https://github.com/Jia-Research-Lab/PFENet.

### :three: Download pre-trained base models

Please download our pre-trained base models from this [google drive link](https://drive.google.com/drive/folders/1OItLbdPTBtu5zZi-mGuHDd53Yx8VSOzE?usp=share_link). Please place the initmodel directory at the `src/` directory of this repo. It contains the pre-trained resnet model. The directories **coco** and **pascal** contains pre-trained base models for different splits of **coco-20i** and **pascal-5i**

## &#x1F5FA; Overview of the repo

Default configuration files can be found in `config/`. The directory `src/lists/` contains the train/val splits for each dataset. All the codes are provided in `src/`. 

## &#x2699; Training The Base 

If you want to train the base models from scratch please run the following: 

```
python3 train_base.py --config=../config/pascal_split0_resnet50_base_m2former.yaml --arch=M2Former  # For pascal-20 split0 base class training
python3 train_base.py --config=../config/coco_split0_resnet50_base_m2former.yaml --arch=M2Former  # For coco-80 split0 base class training
```
Modify the config files accordingly for the split that you want to train. 

## &#x1F9EA; Few-shot fine-tuning

For inductive fine-tuning, please modify the **coco_m2former.yaml** or **pascal_m2former.yaml** (depending on the dataset you want to run inference). Please specify the split and numer of shots you want to evaluate on in the evaluate file, along-with the pre-trained model. 

For transductive fine-tuning, please modify the **coco_m2former_transduction.yaml** or **pascal_m2former_transduction.yaml** in similar manner for the split and number of shots you want to evaluate on. 

To run few-shot inference run first go to `src/` directory and execute **any** of the following commands for inference:

```
python3 test_m2former.py --config ../config/pascal_m2former.yaml  --opts  pi_estimation_strategy self  n_runs 5 gpus [0]  # For pascal inductive inference
python3 test_m2former.py --config ../config/coco_m2former.yaml  --opts  pi_estimation_strategy self  n_runs 5 gpus [0]  # For coco inductive inference
python3 test_m2former.py --config ../config/pascal_m2former_transduction.yaml  --opts  pi_estimation_strategy self  n_runs 5 gpus [0]  # For pascal transductive inference
python3 test_m2former.py --config ../config/coco_m2former_transduction.yaml  --opts  pi_estimation_strategy self  n_runs 5 gpus [0]  # For coco transductive inference

```




## &#x1F64F; Acknowledgments

We thank the authors of [DIaM](https://github.com/sinahmr/DIaM) and [Mask2Former](https://github.com/facebookresearch/Mask2Former) from which some parts of our code are inspired.


## &#x1F4DA; Citation

If you find this project useful, please consider citing:

```bibtex
@inproceedings{hossain2024visual,
  title={Visual Prompting for Generalized Few-shot Segmentation: A Multi-scale Approach},
  author={Hossain, Mir Rayat Imtiaz and Siam, Mennatullah and Sigal, Leonid and Little, James J},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23470--23480},
  year={2024}
}
```
