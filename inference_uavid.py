import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
from catalyst.dl import SupervisedRunner
from skimage.morphology import remove_small_holes, remove_small_objects
from tools.cfg import py2cfg
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from train_supervision import *
import random
import os


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def pv2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 204, 255]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 0, 255]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def landcoverai_to_rgb(mask):
    w, h = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [233, 193, 133]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 255, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def uavid2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [128, 0, 0]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 64, 128]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [0, 128, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [128, 128, 0]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [64, 0, 128]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [192, 0, 192]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [64, 64, 0]
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [0, 0, 0]
    mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--image_path", type=str, default='data/uavid/uavid_test', help="Path to  huge image")
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr"])
    arg("-ph", "--patch-height", help="height of patch size", type=int, default=1152)
    arg("-pw", "--patch-width", help="width of patch size", type=int, default=1024)
    arg("-b", "--batch-size", help="batch size", type=int, default=2)
    arg("-d", "--dataset", help="dataset", default="uavid", choices=["pv", "landcoverai", "uavid"])
    return parser.parse_args()


def load_checkpoint(checkpoint_path, model):
    pretrained_dict = torch.load(checkpoint_path)['model_state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def get_img_padded(image, patch_size):
    oh, ow = image.shape[0], image.shape[1]
    rh, rw = oh % patch_size[0], ow % patch_size[1]

    width_pad = 0 if rw == 0 else patch_size[1] - rw
    height_pad = 0 if rh == 0 else patch_size[0] - rh
    # print(oh, ow, rh, rw, height_pad, width_pad)
    h, w = oh + height_pad, ow + width_pad

    pad = albu.PadIfNeeded(min_height=h, min_width=w, border_mode=0,
                           position='bottom_right', value=[0, 0, 0])(image=image)
    img_pad = pad['image']
    return img_pad, height_pad, width_pad


class InferenceDataset(Dataset):
    def __init__(self, tile_list=None, transform=albu.Normalize()):
        self.tile_list = tile_list
        self.transform = transform

    def __getitem__(self, index):
        img = self.tile_list[index]
        img_id = index
        aug = self.transform(image=img)
        img = aug['image']
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        results = dict(img_id=img_id, img=img)
        return results

    def __len__(self):
        return len(self.tile_list)


def make_dataset_for_one_huge_image(img_path, patch_size):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tile_list = []
    image_pad, height_pad, width_pad = get_img_padded(img.copy(), patch_size)

    output_height, output_width = image_pad.shape[0], image_pad.shape[1]

    for x in range(0, output_height, patch_size[0]):
        for y in range(0, output_width, patch_size[1]):
            image_tile = image_pad[x:x+patch_size[0], y:y+patch_size[1]]
            tile_list.append(image_tile)

    dataset = InferenceDataset(tile_list=tile_list)
    return dataset, width_pad, height_pad, output_width, output_height, image_pad, img.shape


def main():
    args = get_args()
    seed_everything(42)
    seqs = os.listdir(args.image_path)

    # print(img_paths)
    patch_size = (args.patch_height, args.patch_width)
    config = py2cfg(args.config_path)
    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)

    model.cuda(config.gpus[0])
    model.eval()

    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip()
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1, 1.25, 1.5, 1.75]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    for seq in seqs:
        img_paths = []
        output_path = os.path.join(args.output_path, str(seq), 'Labels')
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for ext in ('*.tif', '*.png', '*.jpg'):
            img_paths.extend(glob.glob(os.path.join(args.image_path, str(seq), 'Images', ext)))
        img_paths.sort()
        # print(img_paths)
        for img_path in img_paths:
            img_name = img_path.split('/')[-1]
            # print('origin mask', original_mask.shape)
            dataset, width_pad, height_pad, output_width, output_height, img_pad, img_shape = \
                make_dataset_for_one_huge_image(img_path, patch_size)
            # print('img_padded', img_pad.shape)
            output_mask = np.zeros(shape=(output_height, output_width), dtype=np.uint8)
            output_tiles = []
            k = 0
            with torch.no_grad():
                dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size,
                                        drop_last=False, shuffle=False)
                for input in tqdm(dataloader):
                    # raw_prediction NxCxHxW
                    raw_predictions = model(input['img'].cuda(config.gpus[0]))
                    # print('raw_pred shape:', raw_predictions.shape)
                    raw_predictions = nn.Softmax(dim=1)(raw_predictions)
                    # input_images['features'] NxCxHxW C=3
                    predictions = raw_predictions.argmax(dim=1)
                    image_ids = input['img_id']
                    # print('prediction', predictions.shape)
                    # print(np.unique(predictions))

                    for i in range(predictions.shape[0]):
                        raw_mask = predictions[i].cpu().numpy()
                        mask = raw_mask
                        output_tiles.append((mask, image_ids[i].cpu().numpy()))

            for m in range(0, output_height, patch_size[0]):
                for n in range(0, output_width, patch_size[1]):
                    output_mask[m:m + patch_size[0], n:n + patch_size[1]] = output_tiles[k][0]
                    k = k + 1

            output_mask = output_mask[-img_shape[0]:, -img_shape[1]:]

            # print('mask', output_mask.shape)
            if args.dataset == 'landcoverai':
                output_mask = landcoverai_to_rgb(output_mask)
            elif args.dataset == 'pv':
                output_mask = pv2rgb(output_mask)
            elif args.dataset == 'uavid':
                output_mask = uavid2rgb(output_mask)
            else:
                output_mask = output_mask
            assert img_shape == output_mask.shape
            cv2.imwrite(os.path.join(output_path, img_name), output_mask)


if __name__ == "__main__":
    main()
