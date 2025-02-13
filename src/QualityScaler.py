import functools
import os
import shutil
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from PIL import Image

from QualityScalerUtilities import reverse_split, split_image


def create_temp_dir(name_dir):
    if os.path.exists(name_dir):
        shutil.rmtree(name_dir)
    os.makedirs(name_dir, exist_ok=True)


def find_by_relative_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def image_to_uint(img):
    return img


def save_image(img, img_path):
    img = np.squeeze(img)
    if img.ndim == 3:
        img = img[:, :, [2, 1, 0]]  # Convert BGR to RGB
    cv2.imwrite(img_path, img)


def uint_to_tensor4(img):
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.).unsqueeze(0)


def tensor_to_uint(img):
    img = img.data.squeeze().float().clamp_(0, 1).cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))
    return np.uint8((img * 255.0).round())


def adapt_image_for_deeplearning(img, device):
    backend = torch.device(device if torch.cuda.is_available() else 'cpu')
    img = uint_to_tensor4(image_to_uint(img))
    return img.to(backend, non_blocking=True)


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0.0)


def make_layer(block, n_layers):
    return nn.Sequential(*[block() for _ in range(n_layers)])


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32, bias=True):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv2d(nf + gc * i, gc if i < 4 else nf, 3, 1, 1, bias=bias)
            for i in range(5)
        ])
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        initialize_weights(self.convs, 0.1)

    def forward(self, x):
        features = [x]
        for conv in self.convs[:-1]:
            features.append(self.lrelu(conv(torch.cat(features, 1))))
        return self.convs[-1](torch.cat(features, 1)) * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualDenseBlock_5C(nf, gc) for _ in range(3)])

    def forward(self, x):
        return self.blocks(x) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super().__init__()
        self.sf = sf
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.RRDB_trunk = make_layer(lambda: RRDB(nf, gc), nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        fea = fea + self.trunk_conv(self.RRDB_trunk(fea))
        fea = self.lrelu(self.upconv(F.interpolate(fea, scale_factor=self.sf, mode='nearest')))
        return self.conv_last(self.lrelu(self.HRconv(fea)))


def optimize_torch():
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.profiler.emit_nvtx(False)


def prepare_AI_model(AI_model, device):
    backend = torch.device(device if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join('models', AI_model + '.pth')
    upscale_factor = 2 if "x2" in AI_model else 4 if "x4" in AI_model else 1
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=upscale_factor)
    model.load_state_dict(torch.load(model_path, map_location=backend), strict=True)
    model.eval().to(backend, non_blocking=True)
    return model


def upscale_image(img, model, device):
    img_adapted = adapt_image_for_deeplearning(img, device)
    with torch.no_grad():
        img_upscaled_tensor = model(img_adapted)
        img_upscaled = tensor_to_uint(img_upscaled_tensor)
    return Image.fromarray(img_upscaled)


def process_upscale_image(img_path, AI_model, device):
    model = prepare_AI_model(AI_model, device)
    img = cv2.imread(img_path)
    result_img = upscale_image(img, model, device)
    result_path = img_path.replace('.', '_upscaled.')
    result_img.save(result_path)
    print(f"Upscaling complete. Saved to {result_path}")
    return result_path
