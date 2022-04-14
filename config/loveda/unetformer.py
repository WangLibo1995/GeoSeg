from torch.utils.data import DataLoader
from plseg.losses import *
from plseg.datasets.loveda_dataset import *
from plseg.models.UNetFormer import UNetFormer, EHT2
from catalyst.contrib.nn import Lookahead
from catalyst import utils

# training hparam
max_epoch = 30
ignore_index = len(CLASSES)
train_batch_size = 16
val_batch_size = 16
lr = 6e-4
weight_decay = 0.01
backbone_lr = 6e-5
backbone_weight_decay = 0.01
accumulate_n = 1  # accumulate gradients of 4 batches
num_classes = len(CLASSES)
classes = CLASSES

test_time_aug = 'd4'
output_mask_dir, output_mask_rgb_dir = None, None
weights_name = "unetformer-r18-512crop-ms-epoch30"
weights_path = "model_weights/loveda/{}".format(weights_name)
test_weights_name = "last"
log_name = 'loveda/{}'.format(weights_name)
monitor = 'val_mIoU'
monitor_mode = 'max'
save_top_k = 1
save_last = True
check_val_every_n_epoch = 5
gpus = [0]
strategy = None
pretrained_ckpt_path = None
resume_ckpt_path = None
#  define the network
net = EHT2(num_classes=num_classes, backbone_name='swsl_resnet18', pretrained=True, decode_channels=64)

# define the loss
loss = JointLoss(SoftCrossEntropyLoss(smooth_factor=0.05, ignore_index=ignore_index),
                 DiceLoss(smooth=0.05, ignore_index=ignore_index), 1.0, 1.0)
# loss = UnetFormerLoss(ignore_index=ignore_index)
use_aux_loss = False

# define the dataloader


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # albu.VerticalFlip(p=0.5),
        # albu.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.15),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=512, max_ratio=0.75, ignore_index=ignore_index, nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


train_dataset = LoveDATrainDataset(transform=train_aug, data_root='data/LoveDA/Train_all')

val_dataset = loveda_val_dataset

test_dataset = LoveDATestDataset()

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=train_batch_size,
                          num_workers=4,
                          pin_memory=True,
                          shuffle=True,
                          drop_last=True)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=val_batch_size,
                        num_workers=4,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=False)

# define the optimizer
layerwise_params = {"backbone.*": dict(lr=backbone_lr, weight_decay=backbone_weight_decay)}
net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
base_optimizer = torch.optim.AdamW(net_params, lr=lr, weight_decay=weight_decay)
optimizer = Lookahead(base_optimizer)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=1e-6)

