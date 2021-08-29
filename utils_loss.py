from scipy import misc
import os, cv2, torch
import numpy as np
import torch
import torch.nn as nn
from torchvision import models as visionmodels

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)
    return img

def preprocessing(x):
    x = x/127.5 - 1 # -1 ~ 1
    return x

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):
    return (images+1.) / 2

def imsave(images, size, path):
    return misc.imsave(path, merge(images, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):
    return x.lower() in ('true')

def cam(x, size = 256):
    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    return cam_img / 255.0

def imagenet_norm(x):
    mean = [0.485, 0.456, 0.406]
    std = [0.299, 0.224, 0.225]
    mean = torch.FloatTensor(mean).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    std = torch.FloatTensor(std).unsqueeze(0).unsqueeze(2).unsqueeze(3).to(x.device)
    return (x - mean) / std

def denorm(x):
    return x * 0.5 + 0.5

def tensor2numpy(x):
    return x.detach().cpu().numpy().transpose(1,2,0)

def RGB2BGR(x):
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

def softmask_generator(shadow, shadow_free):
    fake_A2B_ = (shadow_free+1.)/2.
    real_A_   = (shadow+1.)/2.
    diffrAfB  = torch.mean((fake_A2B_-real_A_), dim=1, keepdim=True)
    diffrAfB[diffrAfB<0.05]=0
    mask1crAfB = (diffrAfB - torch.min(diffrAfB)) / (torch.max(diffrAfB) - torch.min(diffrAfB))
    mask1crAfB = mask1crAfB*2-1
    softmask   = torch.cat((mask1crAfB,mask1crAfB,mask1crAfB),dim=1)    #-1.0:non-shadow, 1.0:shadow         
    return softmask

def smooth_loss_masked(pred_map, mask):
    def gradient(pred, mask):
        D_dy      = pred[:, :, 1:] - pred[:, :, :-1]
        mask_D_dy = mask[:, :, 1:] - mask[:, :, :-1]
        D_dx      = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        mask_D_dx = mask[:, :, :, 1:] - mask[:, :, :, :-1]
        return D_dx, D_dy, mask_D_dx, mask_D_dy
    # Loss
    dx,    dy,  mask_dx,   mask_dy = gradient(pred_map, mask)
    loss = (mask_dx*dx).abs().mean()   + (mask_dy*dy).abs().mean()
    return loss

class PerceptualLossVgg16(nn.Module):
    def __init__(self, vgg=None,  gpu_ids=[0,1,2,3], weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg16, self).__init__()
        self.vgg = Vgg16().cuda()
        self.vgg = nn.DataParallel(self.vgg, device_ids=gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0, 1.0, 1.0, 1.0]
        self.indices = indices or [3, 8, 15, 22]
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
        else:
            self.normalize = None
    def forward(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        self.vgg_pretrained_features = visionmodels.vgg16(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                # print(param, param.requires_grad)

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22] # assuming 0 starting index!
        out = []
        #indices = sorted(indices)
        for i in range(indices[-1]+1):
            # print(i, self.vgg_pretrained_features[i])
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out
        
class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

