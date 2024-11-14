import numpy as np
import logging
import os
import torch
import random
import torch.utils.data.sampler as sampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as transforms_f
from PIL import Image
from PIL import ImageFilter
import matplotlib.pyplot as plt

def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])
    
    elif dataset == 'oucuavseg':
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([144, 238, 144])
        cmap[2] = np.array([0, 100, 0])
        cmap[3] = np.array([85, 85, 85])
        cmap[4] = np.array([0, 102, 204])
        cmap[5] = np.array([0, 153, 153])
        cmap[6] = np.array([128, 113, 82])
        cmap[7] = np.array([139, 69, 19])
        cmap[8] = np.array([128, 119, 177])
        cmap[9] = np.array([255, 192, 203])
        cmap[10] = np.array([205, 133, 63])
        cmap[11] = np.array([124, 252, 0])
        cmap[12] = np.array([192, 192, 192])
        cmap[13] = np.array([173, 216, 230])
    
    elif dataset == 'tas500':
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([192, 192, 192])
        cmap[2] = np.array([106, 105, 105])
        cmap[3] = np.array([160, 82, 45])
        cmap[4] = np.array([245, 164, 96])
        cmap[5] = np.array([61, 179, 113])
        cmap[6] = np.array([34, 139, 34])
        cmap[7] = np.array([154, 205, 50])
        cmap[8] = np.array([1, 128, 0])
        cmap[9] = np.array([61, 179, 113])
        cmap[10] = np.array([1, 220, 124])
        cmap[11] = np.array([139, 69, 19])
        cmap[12] = np.array([2, 51, 73])
        cmap[13] = np.array([191, 153, 153])
        cmap[14] = np.array([1, 132, 111])
        cmap[15] = np.array([1, 0, 142])
        cmap[16] = np.array([1, 60, 100])
        cmap[17] = np.array([70, 130, 180])
        cmap[18] = np.array([128, 0, 128])
        cmap[19] = np.array([154, 153, 153])
        cmap[20] = np.array([221, 220, 0])
        cmap[21] = np.array([221, 20, 60])
        cmap[22] = np.array([255, 182, 193])
        cmap[23] = np.array([221, 220, 220])

    elif dataset == 'loveda':
        cmap[0] = np.array([0, 0, 0])
        cmap[1] = np.array([253, 252, 252])
        cmap[2] = np.array([223, 49, 33])
        cmap[3] = np.array([255, 255, 85])
        cmap[4] = np.array([1, 2, 230])
        cmap[5] = np.array([151, 126, 178])
        cmap[6] = np.array([114, 247, 76])
        cmap[7] = np.array([247, 198, 131])
    
    return cmap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explicit requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()


def init_log(name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    if "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        logger.addFilter(lambda record: rank == 0)
    else:
        rank = 0
    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def transform(image, label, logits=None, crop_size=(512, 512), scale_size=(0.8, 1.0), augmentation=True):
    # Random rescale image
    raw_w, raw_h = image.size
    scale_ratio = random.uniform(scale_size[0], scale_size[1])

    resized_size = (int(raw_h * scale_ratio), int(raw_w * scale_ratio))
    image = transforms_f.resize(image, resized_size, Image.BILINEAR)
    label = transforms_f.resize(label, resized_size, Image.NEAREST)
    if logits is not None:
        logits = transforms_f.resize(logits, resized_size, Image.NEAREST)

    # Add padding if rescaled image size is less than crop size
    if crop_size == -1:  # use original im size without crop or padding
        crop_size = (raw_h, raw_w)

    if crop_size[0] > resized_size[0] or crop_size[1] > resized_size[1]:
        right_pad, bottom_pad = max(crop_size[1] - resized_size[1], 0), max(crop_size[0] - resized_size[0], 0)
        image = transforms_f.pad(image, padding=(0, 0, right_pad, bottom_pad), padding_mode='reflect')
        label = transforms_f.pad(label, padding=(0, 0, right_pad, bottom_pad), fill=255, padding_mode='constant')
        if logits is not None:
            logits = transforms_f.pad(logits, padding=(0, 0, right_pad, bottom_pad), fill=0, padding_mode='constant')

    # Random Cropping
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=crop_size)
    image = transforms_f.crop(image, i, j, h, w)
    label = transforms_f.crop(label, i, j, h, w)
    if logits is not None:
        logits = transforms_f.crop(logits, i, j, h, w)

    if augmentation:
        # Random color jitter
        if torch.rand(1) > 0.2:
            #  color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))  For PyTorch 1.9/TorchVision 0.10 users
            color_transform = transforms.ColorJitter((0.75, 1.25), (0.75, 1.25), (0.75, 1.25), (-0.25, 0.25))
            image = color_transform(image)

        # Random Gaussian filter
        if torch.rand(1) > 0.5:
            sigma = random.uniform(0.15, 1.15)
            image = image.filter(ImageFilter.GaussianBlur(radius=sigma))

        # Random horizontal flipping
        if torch.rand(1) > 0.5:
            image = transforms_f.hflip(image)
            label = transforms_f.hflip(label)
            if logits is not None:
                logits = transforms_f.hflip(logits)

    # Transform to tensor
    image = transforms_f.to_tensor(image)
    label = (transforms_f.to_tensor(label) * 255).long()
    label[label == 255] = -1  # invalid pixels are re-mapped to index -1
    if logits is not None:
        logits = transforms_f.to_tensor(logits)

    # Apply (ImageNet) normalisation
    image = transforms_f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if logits is not None:
        return image, label, logits
    else:
        return image, label


#****************#
def denormalise(x, imagenet=True):
    if imagenet:
        x = transforms_f.normalize(x, mean=[0., 0., 0.], std=[1 / 0.229, 1 / 0.224, 1 / 0.225])
        x = transforms_f.normalize(x, mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.])
        return x
    else:
        return (x + 1) / 2
    
def tensor_to_pil(im, label, logits):
    im = denormalise(im)
    im = transforms_f.to_pil_image(im.cpu())

    label = label.float() / 255.
    label = transforms_f.to_pil_image(label.unsqueeze(0).cpu())

    logits = transforms_f.to_pil_image(logits.unsqueeze(0).cpu())
    return im, label, logits


def batch_transform(data, label, logits, crop_size, scale_size, apply_augmentation):
    data_list, label_list, logits_list = [], [], []
    device = data.device

    for k in range(data.shape[0]):
        data_pil, label_pil, logits_pil = tensor_to_pil(data[k], label[k], logits[k])
        aug_data, aug_label, aug_logits = transform(data_pil, label_pil, logits_pil,
                                                    crop_size=crop_size,
                                                    scale_size=scale_size,
                                                    augmentation=apply_augmentation)
        data_list.append(aug_data.unsqueeze(0))
        label_list.append(aug_label)
        logits_list.append(aug_logits)

    data_trans, label_trans, logits_trans = \
        torch.cat(data_list).to(device), torch.cat(label_list).to(device), torch.cat(logits_list).to(device)
    return data_trans, label_trans, logits_trans






# --------------------------------------------------------------------------------
# Define semi-supervised methods (based on data augmentation)
# --------------------------------------------------------------------------------
def generate_cutout_mask(img_size, ratio=2):
    cutout_area = img_size[0] * img_size[1] / ratio

    w = np.random.randint(img_size[1] / ratio + 1, img_size[1])
    h = np.round(cutout_area / w)

    x_start = np.random.randint(0, img_size[1] - w + 1)
    y_start = np.random.randint(0, img_size[0] - h + 1)

    x_end = int(x_start + w)
    y_end = int(y_start + h)

    mask = torch.ones(img_size)
    mask[y_start:y_end, x_start:x_end] = 0
    return mask.float()


def generate_class_mask(pseudo_labels):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_unsup_data(data, target, logits, mode='cutout'):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        if mode == 'cutout':
            mix_mask = generate_cutout_mask([im_h, im_w], ratio=2).to(device)
            target[i][(1 - mix_mask).bool()] = -1

            new_data.append((data[i] * mix_mask).unsqueeze(0))
            new_target.append(target[i].unsqueeze(0))
            new_logits.append((logits[i] * mix_mask).unsqueeze(0))
            continue

        if mode == 'cutmix':
            mix_mask = generate_cutout_mask([im_h, im_w]).to(device)
        if mode == 'classmix':
            mix_mask = generate_class_mask(target[i]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long(), new_logits

