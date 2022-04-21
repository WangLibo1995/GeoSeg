[Welcome to my homepage!](https://WangLibo1995.github.io)

## Introduction

**GeoSeg** is an open-source  semantic segmentation toolbox based on PyTorch, [pytorch lightning](https://www.pytorchlightning.ai/) and [timm](https://github.com/rwightman/pytorch-image-models), 
which mainly focuses on using advanced CNNs and Vision Transformers for remote sensing image segmentation.


## Major Features

- Unified Benchmark

  we provide a unified training script for various segmentation methods.
  
- Simple and Effective

  Thanks to **pytorch lightning** and **timm** , the code is easy for further development.
  
- Support More Remote Sensing Datasets
 
  - [ISPRS Vaihingen and Potsdam](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx) 
  - [UAVid](https://uavid.nl/)
  - [LoveDA](https://codalab.lisn.upsaclay.fr/competitions/421)
  - [WHU building](http://gpcv.whu.edu.cn)
  - [Inria Aerial Image Labelling](https://project.inria.fr/aerialimagelabeling/)
  - More datasets will be supported in the future.
  
- Multi-scale Training and Testing
- Inference on Huge Remote Sensing Images

## Supported Networks

- Vision Transformer

  - [UNetFormer](http://arxiv.org/abs/2109.08937) 
  - [DC-Swin](https://ieeexplore.ieee.org/abstract/document/9681903)
  - [BANet](https://www.mdpi.com/2072-4292/13/16/3065)
  
- CNN
 
  - [MANet](https://ieeexplore.ieee.org/abstract/document/9487010) 
  - [ABCNet](https://www.sciencedirect.com/science/article/pii/S0924271621002379)
  - [A2FPN](https://www.tandfonline.com/doi/full/10.1080/01431161.2022.2030071)
  
## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── GeoSeg (code)
├── pretrain_weights (save the pretrained weights like vit, swin, etc)
├── model_weights (save the model weights)
├── fig_results (save the masks predicted by models)
├── lightning_logs (CSV format training logs)
├── data
│   ├── LoveDA
│   │   ├── Train
│   │   │   ├── Urban
│   │   │   │   ├── images_png (original images)
│   │   │   │   ├── masks_png (original masks)
│   │   │   │   ├── masks_png_convert (converted masks used for training)
│   │   │   │   ├── masks_png_convert_rgb (original rgb format masks)
│   │   │   ├── Rural
│   │   │   │   ├── images_png 
│   │   │   │   ├── masks_png 
│   │   │   │   ├── masks_png_convert
│   │   │   │   ├── masks_png_convert_rgb
│   │   ├── Val (the same with Train)
│   │   ├── test
│   ├── uavid
│   │   ├── uavid_train (original)
│   │   ├── uavid_val (original)
│   │   ├── uavid_test (original)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   ├── vaihingen
│   │   ├── train_images (original)
│   │   ├── train_masks (original)
│   │   ├── train_masks_eroded (original)
│   │   ├── val_images (original)
│   │   ├── val_masks (original)
│   │   ├── val_masks_eroded (original)
│   │   ├── test_images (original)
│   │   ├── test_masks (original)
│   │   ├── test_masks_eroded (original)
│   │   ├── train (processed)
│   │   ├── val (processed)
│   │   ├── test (processed)
│   ├── potsdam (the same with vaihingen)
```

## Install

Open the folder **airs** using **Linux Terminal** and create python environment:
```
conda create -n airs python=3.8
conda activate airs

conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r GeoSeg/requirements.txt
```
## Data Preprocessing

**LoveDA**
```
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Rural/masks_png --output-mask-dir data/LoveDA/Train/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Train/Urban/masks_png --output-mask-dir data/LoveDA/Train/Urban/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Rural/masks_png --output-mask-dir data/LoveDA/Val/Rural/masks_png_convert
python GeoSeg/tools/loveda_mask_convert.py --mask-dir data/LoveDA/Val/Urban/masks_png --output-mask-dir data/LoveDA/Val/Urban/masks_png_convert
```
More datasets are updating.

## Training

**LoveDA**
```
python GeoSeg/train_supervision.py -c GeoSeg/config/loveda/unetformer.py
```

## Validation

**LoveDA**
```
python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/unetformer.py -o fig_results/loveda/unetformer_val --rgb --val -t 'd4'
```

## Testing

**LoveDA**
```
python GeoSeg/loveda_test.py -c GeoSeg/config/loveda/unetformer.py -o fig_results/loveda/unetformer_test --rgb -t 'd4'
```
## Citation

If you find this project useful in your research, please consider cite [our papers](https://WangLibo1995.github.io).


## Acknowledgement

We wish **GeoSeg** could serve the growing research of remote sensing by providing a unified benchmark 
and inspiring researchers to develop their own segmentation networks. Many thanks the following projects's contributions to **GeoSeg**.
- [pytorch lightning](https://www.pytorchlightning.ai/)
- [timm](https://github.com/rwightman/pytorch-image-models)
- [pytorch-toolbelt](https://github.com/BloodAxe/pytorch-toolbelt)
- [ttach](https://github.com/qubvel/ttach)
- [catalyst](https://github.com/catalyst-team/catalyst)
- [mmsegmentation](https://github.com/open-mmlab/mmsegmentation)
