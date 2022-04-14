import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import glob
from PIL import Image
import ttach as tta
import cv2
import numpy as np
import torch
import albumentations as albu
import random

from skimage.morphology import remove_small_holes, remove_small_objects

from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)


def remove_small_objects_and_holes(label, min_size, area_threshold, in_place=False):
    label = label.astype(np.bool)
    label = remove_small_objects(label, min_size=min_size, connectivity=1, in_place=in_place)
    label = remove_small_holes(label, area_threshold=area_threshold, connectivity=1, in_place=in_place)
    label = label.astype(np.uint8)
    return label


def post_process(mask, min_size=None, min_area=None, num_classes=6):
    mask_one_hot = np.stack([(mask == v) for v in range(num_classes)], axis=-1)
    # mask_one_hot = mask_one_hot.transpose(2, 0, 1)
    # print('mask_one_hot shape:', mask_one_hot.shape)
    for i in range(num_classes):
        single_class_one_hot = mask_one_hot[:, :, i].copy()
        mask_one_hot[:, :, i] = remove_small_objects_and_holes(
                                       label=single_class_one_hot,
                                       min_size=min_size, area_threshold=min_area)
    mask_new = torch.from_numpy(mask_one_hot).float()
    mask_new = mask_new.argmax(dim=2)
    # # print('mask_new shape:', mask_new)
    mask = mask_new.cpu().numpy().astype(np.uint8)
    # mask = mask_one_hot.astype(np.uint16)
    # print('final mask shape:', mask.shape)
    return mask


def label2rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [255, 0, 0]
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [255, 255, 0]
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [0, 0, 255]
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [159, 129, 183]
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [0, 255, 0]
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 195, 128]
    return mask_rgb


def img_writer(inp):
    (mask,  mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label2rgb(mask)
        mask_tif = cv2.cvtColor(mask_tif, cv2.COLOR_RGB2BGR)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default=None, choices=[None, "d4", "lr"])
    arg("-m", "--min-size", help="small obj area thread", type=int, default=None)
    arg("--rgb", help="whether output rgb images", action='store_true')
    arg("--val", help="whether eval validation set", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(os.path.join(config.weights_path, config.test_weights_name+'.ckpt'), config=config)
    model.cuda()
    model.eval()
    if args.tta == "lr":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                # tta.Scale(scales=[1, 2, 4]),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                # tta.VerticalFlip(),
                # tta.Rotate90(angles=[0, 90, 180, 270]),
                tta.Scale(scales=[0.75, 1.0, 1.25, 1.5], interpolation='bicubic', align_corners=False),
                # tta.Multiply(factors=[0.8, 1, 1.2])
            ]
        )
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset
    if args.val:
        evaluator = Evaluator(num_class=config.num_classes)
        evaluator.reset()
        test_dataset = config.val_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=2,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        # crf = DenseCRF(iter_max=10, pos_xy_std=3, pos_w=3, bi_xy_std=140, bi_rgb_std=5, bi_w=5)
        for input in tqdm(test_loader):
            # raw_prediction NxCxHxW
            raw_predictions = model(input['img'].cuda())
            # print('raw_pred shape:', raw_predictions.shape)
            # input_images['features'] NxCxHxW C=3

            image_ids = input["img_id"]
            if args.val:
                masks_true = input['gt_semantic_seg']

            img_type = input['img_type']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            # input_images['features'] NxCxHxW C=3
            predictions = raw_predictions.argmax(dim=1)
            # print('preds shape', predictions[0,:,:])

            for i in range(raw_predictions.shape[0]):
                raw_mask = predictions[i].cpu().numpy()
                # mask = raw_mask
                if args.min_size:
                    mask = post_process(mask=raw_mask, min_size=args.min_size,
                                        min_area=args.min_size, num_classes=config.num_classes)
                else:
                    mask = raw_mask
                # print(mask.shape)
                mask_name = image_ids[i]
                mask_type = img_type[i]
                if args.val:
                    if not os.path.exists(os.path.join(args.output_path, mask_type)):
                        os.mkdir(os.path.join(args.output_path, mask_type))
                    evaluator.add_batch(pre_image=mask, gt_image=masks_true[i].cpu().numpy())
                    results.append((mask, str(args.output_path / mask_type / mask_name), args.rgb))
                else:
                    results.append((mask, str(args.output_path / mask_name), args.rgb))
    if args.val:
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        for class_name, class_iou, class_f1 in zip(config.classes, iou_per_class, f1_per_class):
            print('F1_{}:{}, IOU_{}:{}'.format(class_name, class_f1, class_name, class_iou))
        print('F1:{}, mIOU:{}, OA:{}'.format(np.nanmean(f1_per_class), np.nanmean(iou_per_class), OA))

    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    img_write_time = t1 - t0
    print('images writing spends: {} s'.format(img_write_time))


if __name__ == "__main__":
    main()
