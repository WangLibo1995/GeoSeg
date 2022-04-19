[Welcome to my homepage!](https://WangLibo1995.github.io)

## Folder Structure

Prepare the following folders to organize this repo:
```none
airs
├── geovision_transformer (code)
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

## Prerequisites

- Linux
- Python 3.7+
- PyTorch 1.10
- torchvision 0.11.1
- pytorch-lightning 1.5.9
- timm 0.5.4
- catalyst 20.09
- albumentations 1.1.0
- opencv-python 4.5.4.60
- other packages