from __future__ import print_function
import os
import datetime
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import utils as vutils
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import cv2
from torchvision import models as visionmodels
import torch.nn.functional as F
import numpy as np
from torchvision import transforms as transforms
from torchvision import utils as vutils
from PIL import Image
from models.networks import Vgg16, MeanShift
import matplotlib.pyplot as plt
torch.manual_seed(17)

dataroot_A = './input/'
dataroot_C = './output/'
im_suf_A = '.png'
im_suf_C = '.png'
# im_suf_A = '.jpg'
# im_suf_C = '.jpg'
out_list = [os.path.splitext(f)[0] for f in os.listdir(dataroot_A) if f.endswith(im_suf_A)]

total_avg_rmse = 0.0
all_avg_rmse = 0.0

def main():
    # Get the VGG16 network
    nets_vgg      = Vgg16().cuda()
    nets_vgg_norm = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).cuda()
    # Get data
    total_cumulative_rmse = 0
    total_cur_rmse = 0 

    avg_rmse = 0.0 
    num = 21
    num_layer       = [num+1]
    print('num_layer',num_layer) ##num_layer 22
    all_cumulative_rmse = 0
    all_cur_rmse = 0 

    for idx, img_name in enumerate(out_list):
        print('predicting: %d / %d' % (idx + 1, len(out_list)))
        img_A_path = os.path.join(dataroot_A, img_name + im_suf_A)
        # img_nameC = img_name[:-4]+'noshad'
        # img_C_path = os.path.join(dataroot_C, img_nameC + im_suf_C)  #this is for noshad name
        img_C_path = os.path.join(dataroot_C, img_name + im_suf_C)   #this is the same name
        in1_shadow_g     = process_img(img_A_path)
        in1_shadow_g     = (in1_shadow_g+1)*0.5
        out1_free_g      = process_img(img_C_path)
        out1_free_g      = (out1_free_g+1)*0.5
        # Prepare data for VGG
        in1_shadow_vgg  = nets_vgg_norm(in1_shadow_g)
        out1_free_vgg   = nets_vgg_norm(out1_free_g)
        in1_shadow_fts  = nets_vgg(in1_shadow_vgg, num_layer)
        out1_shadow_fts = nets_vgg(out1_free_vgg, num_layer)
        # Save checks dir
        results_name   = 'shadow_VGGfeatures/'
        savechecks_dir = './results_VGGfeatures/' + results_name + str(num+1) + '/' + img_name + '/'
        os.makedirs(savechecks_dir, exist_ok=True)
        savechecks_dir_in  = savechecks_dir + 'input_feature/'
        savechecks_dir_out = savechecks_dir + 'output_feature/'
        os.makedirs(savechecks_dir_in, exist_ok=True)
        os.makedirs(savechecks_dir_out, exist_ok=True)
        assert(len(in1_shadow_fts)==1)
        in1_shadow_fts  = in1_shadow_fts[0]
        out1_shadow_fts = out1_shadow_fts[0]
        num_fts         = in1_shadow_fts.size()[1]
        cumulative_rmse = 0
        cur_rmse = 0
        for num_ft in range(num_fts):
            size_input    = (in1_shadow_g.size()[2], in1_shadow_g.size()[3])
            in1_shadow_ft = F.interpolate((in1_shadow_fts[:, num_ft:num_ft+1, :, :]), 
                                         size_input)
            out_free_ft    = F.interpolate((out1_shadow_fts[:,  num_ft:num_ft+1, :, :]), 
                                         size_input)
            in1_shadow_ft  = torch.div(in1_shadow_ft - torch.min(in1_shadow_ft), 
                                     torch.max(in1_shadow_ft) - torch.min(in1_shadow_ft))
            out_free_ft    = torch.div(out_free_ft - torch.min(out_free_ft), 
                                     torch.max(out_free_ft) - torch.min(out_free_ft))
            cmap          = plt.get_cmap('jet')
            in1_shadow    = in1_shadow_g.detach().cpu()
            out1_free     = out1_free_g.detach().cpu()
            in1_shadow_ft_c = torch.FloatTensor(cmap(in1_shadow_ft[0, 0, :, :].detach().cpu().numpy())[:, :, :-1]).permute(2,0,1).unsqueeze(0) 
            out_free_ft_c   = torch.FloatTensor(cmap(out_free_ft[0, 0, :, :].detach().cpu().numpy())[:, :, :-1]).permute(2,0,1).unsqueeze(0)
            cur_rmse = torch.mean((in1_shadow_ft_c-out_free_ft_c)**2)**0.5 
            cumulative_rmse += cur_rmse
            avg_rmse = cumulative_rmse / (num_ft+1)
            saveimg       = torch.cat((in1_shadow, in1_shadow_ft_c, out1_free, out_free_ft_c), dim=3)
            savename      = savechecks_dir + '/visual_' + '%03d'%num_ft + '_'+ str('%.4f'%cur_rmse) +'.jpg'
            vutils.save_image(saveimg, savename)
            ft_in = cv2.cvtColor((cmap(in1_shadow_ft[0, 0, :, :].detach().cpu().numpy())[:, :, :-1]*255).astype('float32'), cv2.COLOR_BGR2RGB)
            ft_out = cv2.cvtColor((cmap(out_free_ft[0, 0, :, :].detach().cpu().numpy())[:, :, :-1]*255).astype('float32'),cv2.COLOR_BGR2RGB)
            cv2.imwrite((savechecks_dir_in + '/in_' + '%03d'%num_ft + '_'+ str('%.4f'%cur_rmse) +'.jpg'), ft_in)
            cv2.imwrite((savechecks_dir_out + '/out_' + '%03d'%num_ft + '_'+ str('%.4f'%cur_rmse) +'.jpg'), ft_out)
        print('RSME for every img is %.4f'%(avg_rmse)) 
        all_cumulative_rmse += avg_rmse
    total_avg_rmse = total_cumulative_rmse / (num+1)

# Some functions here
def gray2rgb(inimg):
    outimg = torch.cat((inimg, inimg, inimg), 1)
    return outimg

def normalize():
    return transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
def inv_normalize():
    return transforms.Normalize(mean=[-0.5 / 0.5, -0.5 / 0.5, -0.5 / 0.5],
                                std=[1 / 0.5, 1 / 0.5, 1 / 0.5])
def process_img(fname):
    # Setup functions
    norm_  = normalize()
    totens_= transforms.ToTensor()
    # Load and normalize images
    imgL_o = Image.open(fname).convert('RGB')
    imgL   = norm_(totens_(imgL_o)).numpy()
    imgL   = torch.FloatTensor(np.reshape(imgL, [1, 3, imgL.shape[1], imgL.shape[2]])).cuda()
    return imgL

class PerceptualLossVgg16(nn.Module):
    def __init__(self, vgg=None, weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg16, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16().cuda()
        else:
            self.vgg = vgg
        # self.vgg = nn.DataParallel(self.vgg, device_ids=gpu_ids)
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

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22] # assuming 0 starting index!
        out = []
        #indices = sorted(indices)
        for i in range(indices[-1]+1):
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

if __name__ == '__main__':
    main()