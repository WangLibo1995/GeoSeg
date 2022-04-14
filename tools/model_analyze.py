from tools.torchstat import stat
from ptflops import get_model_complexity_info
from plseg.models.UNetFormer import EHT2
from plseg.models.SwinUNet import SwinUNet
from plseg.models.TransUNet import TransUNet
from plseg.models.UNet import UNet
from plseg.models.UNetformerDA import UNetFormerDA
from plseg.models.UNetformerCC import UNetFormerCC
from plseg.models.UNetformerLA import UNetFormerLA
from plseg.models.UNetformerPA import UNetFormerPA
import torch
from torchvision.models.resnet import resnet18, resnet34, resnet50, resnet101
import numpy as np

if __name__ == '__main__':
    # net = TransUNet(num_classes=8, pretrained=False, backbone_name='resnet50')
    # net = UNetFormerLA(num_classes=8, pretrained=False, backbone_name='resnet18', decode_channels=96)
    net = UNet(bilinear=False)
    # net = SwinUNet(img_size=512, pretrained_path=None)
    macs, params = get_model_complexity_info(net, (3, 512, 512), as_strings=True,
                                             print_per_layer_stat=False, verbose=False)
    print('Computational complexity: ', macs)
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print(net.swin_unet.flops())
    # net = LTSeg(pretrained=True, weight_path='lt_lite.pth')
    # print("Parameters : {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
    # print(stat(net, (3, 512, 512)))
    # model = resnet50()
    # print("Parameters : {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    # a = np.zeros([2,2,3])
    # a[np.all(a==[0,0,0], axis=-1)] = [1, 1, 1]
    # print(a)