import os.path
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import random

from PIL import Image


class TestDataset(Dataset):
    def __init__(self, opt):
        super(TestDataset, self).__init__()

        self.opt = opt
        self.root = opt.input_path
        self.mode = opt.phase

        self.fine_height=256
        self.fine_width=192

        self.text = '/home/woo/Desktop/job/project/VITON/BASILIUM-AI/model/test_pairs.txt'

        self.image_dir = os.path.join(opt.input_path, opt.phase + '_img')
        self.clothes_dir = os.path.join(opt.input_path, opt.phase + '_clothes')

        self.image_path = []
        self.clothes_path = []
        self.get_file_name()
        self.dataset_size = len(self.image_path)


    def get_file_name(self):
        with open(self.text, 'r') as f:
            for line in f.readlines():
                image_name, clothes_name = line.strip().split()
                self.image_path.append(os.path.join(self.image_dir, image_name))
                self.clothes_path.append(os.path.join(self.clothes_dir, clothes_name))


    def __getitem__(self, index):        
        image = cv2.imread(self.image_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        params = get_params(self.opt, image.shape[0], image.shape[1])
        transform = get_transform(self.opt, params)
        transform_edge = get_transform(self.opt, params, method=cv2.INTER_AREA, normalize=False)

        image_tensor = transform(image)

        cloth = cv2.imread(self.clothes_path[index])
        cloth = cv2.cvtColor(cloth , cv2.COLOR_BGR2RGB)
        cloth_tensor = transform(cloth)

        edge = cv2.cvtColor(cloth, cv2.COLOR_BGR2GRAY)
        _, edge = cv2.threshold(edge, 245, 255, cv2.THRESH_BINARY_INV)
        edge_tensor = transform_edge(edge)

        input_dict = { 'image': image_tensor,'clothes': cloth_tensor, 'edge': edge_tensor, 'p_name':self.image_path[index].split('/')[-1]}

        return input_dict

    def __len__(self):
        return self.dataset_size 

    def name(self):
        return 'AlignedDataset'
    
    def get_image_shape(self):
        image = cv2.imread(self.image_path[0])
        return image.shape
    

def get_params(opt, h, w):
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))

    flip = 0
    # return {'crop_pos': (x, y), 'flip': flip}
    return {'crop_pos': (y, x), 'flip': flip}


def get_transform_resize(opt, params, method=cv2.INTER_AREA, normalize=True):
    transform_list = []
    transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
    osize = [192, 256]
    transform_list.append(transforms.Scale(osize, method))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def get_transform(opt, params, method=cv2.INTER_AREA, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Lambda(lambda img: cv2.resize(img, (osize[0], osize[1]), interpolation=method)))
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        osize = [192, 256]
        transform_list.append(transforms.Lambda(lambda img: cv2.resize(img, (osize[0], osize[1]), interpolation=method)))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(16)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.shape[1], img.shape[0]    
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return cv2.resize(img, (h, w), interpolation=method)

def __scale_width(img, target_width, method=cv2.INTER_AREA):
    oh, ow = img.shape[0], img.shape[1]
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow) 
    return cv2.resize(img, (h, w), interpolation=method)

def __crop(img, pos, size):
    oh, ow = img.shape[0], img.shape[1]
    # x1, y1 = pos
    y1, x1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        # return img[x1:x1+tw, y1:y1+th, :]
        return img[y1:y1+th, x1:x1+tw, :]
    return img

def __flip(img, flip):
    if flip:
        return cv2.flip(img, 1)
    return img
