import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
import functools, itertools
import numpy as np
from util.util import gkern_2d
import os
from torchvision import models as visionmodels

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        return functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        return functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)


def define_G(input_nc, output_nc, ngf, n_blocks, n_blocks_shared, n_domains, norm='batch', use_dropout=False, gpu_ids=[]):
    norm_layer = get_norm_layer(norm_type=norm)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    n_blocks -= n_blocks_shared
    n_blocks_enc = n_blocks // 2
    n_blocks_dec = n_blocks - n_blocks_enc

    dup_args = (ngf, norm_layer, use_dropout, gpu_ids, use_bias)
    enc_args = (input_nc, n_blocks_enc) + dup_args
    dec_args = (output_nc, n_blocks_dec) + dup_args

    if n_blocks_shared > 0:
        n_blocks_shdec = n_blocks_shared // 2
        n_blocks_shenc = n_blocks_shared - n_blocks_shdec
        shenc_args = (n_domains, n_blocks_shenc) + dup_args
        shdec_args = (n_domains, n_blocks_shdec) + dup_args
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args, ResnetGenShared, shenc_args, shdec_args)
    else:
        plex_netG = G_Plexer(n_domains, ResnetGenEncoder, enc_args, ResnetGenDecoder, dec_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netG.cuda(gpu_ids[0])

    plex_netG.apply(weights_init)
    return plex_netG


def define_D(input_nc, ndf, netD_n_layers, n_domains, tensor, 
             norm='batch', gpu_ids=[],disc_model='NLayerDiscriminator'):
    norm_layer = get_norm_layer(norm_type=norm)

    model_args = (input_nc, ndf, netD_n_layers, tensor, norm_layer, gpu_ids)

    if disc_model == 'NLayerDiscriminator_RGBonly':
        plex_netD = D_Plexer(n_domains, NLayerDiscriminator_RGBonly, model_args)
    else:
        # Default is 'NLayerDiscriminator' in TodayGAN
        plex_netD = D_Plexer(n_domains, NLayerDiscriminator, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netD.cuda(gpu_ids[0])

    plex_netD.apply(weights_init)
    return plex_netD

def define_DispNet(maxdisp, model_dispnet, use_grayscale_images, n_domains, gpu_ids=[]):
    model_args      =(model_dispnet, maxdisp, use_grayscale_images, gpu_ids)
    plex_netDispNet = DispNet_Plexer(n_domains, PSMNet_model_, model_args)

    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        plex_netDispNet.cuda(gpu_ids[0])

    plex_netDispNet.apply(weights_init)
    return plex_netDispNet

##############################################################################
# Losses
##############################################################################
# Defines the GAN loss which uses the Relativistic LSGAN
def GANLoss(inputs_real, inputs_fake, is_discr):
    if is_discr:
        y = -1
    else:
        y = 1
        inputs_real = [i.detach() for i in inputs_real]
    loss = lambda r,f : torch.mean((r-f+y)**2)
    losses = [loss(r,f) for r,f in zip(inputs_real, inputs_fake)]
    multipliers = list(range(1, len(inputs_real)+1));  multipliers[-1] += 1
    losses = [m*l for m,l in zip(multipliers, losses)]
    return sum(losses) / (sum(multipliers) * len(losses))

##############################################################################
# Classes
##############################################################################
# A slight modification to parallelize the data before inputting to PSMNet
class PSMNet_model_(nn.Module):
    def __init__(self, model_PSMNet, maxdisp, use_grayscale_images, gpu_ids=[]):
        super(PSMNet_model_, self).__init__()
        self.maxdisp = maxdisp
        if model_PSMNet == 'PSMNet_stackhourglass':
            from .PSMNet_stackhourglass import PSMNet as PSMNet_stackhourglass
            self.module  = PSMNet_stackhourglass(self.maxdisp, use_grayscale_images)
        elif model_PSMNet == 'PSMNet_basic':
            from .PSMNet_basic import PSMNet as PSMNet_basic
            self.module  = PSMNet_basic(self.maxdisp, use_grayscale_images)
        self.gpu_ids = gpu_ids

    def forward(self, left, right):
        if self.gpu_ids and isinstance(left.data, torch.cuda.FloatTensor) and isinstance(right.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.module, (left, right), self.gpu_ids)
        return self.module(left, right)

# Defines the generator that consists of Resnet blocks between a few downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenEncoder(nn.Module):
    def __init__(self, input_nc, n_blocks=4, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.PReLU()]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.PReLU()]

        mult = 2**n_downsampling
        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)

class ResnetGenShared(nn.Module):
    def __init__(self, n_domains, n_blocks=2, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenShared, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, n_domains=n_domains,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        self.model = SequentialContext(n_domains, *model)

    def forward(self, input, domain):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, (input, domain), self.gpu_ids)
        return self.model(input, domain)

class ResnetGenDecoder(nn.Module):
    def __init__(self, output_nc, n_blocks=5, ngf=64, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, gpu_ids=[], use_bias=False, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenDecoder, self).__init__()
        self.gpu_ids = gpu_ids

        model = []
        n_downsampling = 2
        mult = 2**n_downsampling

        for _ in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias, padding_type=padding_type)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=4, stride=2,
                                         padding=1, output_padding=0,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.PReLU()]

        model += [nn.ReflectionPad2d(3),
                  nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, norm_layer, use_dropout, use_bias, padding_type='reflect', n_domains=0):
        super(ResnetBlock, self).__init__()

        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.PReLU()]
        if use_dropout:
            print('Adding dropout layers in the network!')
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim + n_domains, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        self.conv_block = SequentialContext(n_domains, *conv_block)

    def forward(self, input):
        if isinstance(input, tuple):
            return input[0] + self.conv_block(*input)
        return input + self.conv_block(input)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
                 tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids     = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.dsamp_filter= tensor([1]).view(1,1,1,1)
        self.blur_filter = tensor(gkern_2d())
        self.input_nc    = input_nc

        self.model_rgb  = self.model(input_nc, ndf, n_layers, norm_layer)
        self.model_gray = self.model(1, ndf, n_layers, norm_layer)
        self.model_grad = self.model(2, ndf, n_layers-1, norm_layer)

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        if self.input_nc == 1:
            # Re-adjust the Gaussian filter if the images are in grayscale
            self.blur_filter = self.blur_filter[0:1, :, :, :]
            blurred_left  = torch.nn.functional.conv2d(input[:, 0:1, :, :], self.blur_filter, groups=1, padding=2)
            blurred_right = torch.nn.functional.conv2d(input[:, 1:2, :, :], self.blur_filter, groups=1, padding=2)
            gray_left     = input[:, 0:1, :, :]
            gray_right    = input[:, 1:2, :, :]
        elif self.input_nc == 3:
            blurred_left = torch.nn.functional.conv2d(input[:, 0:3, :, :], self.blur_filter, groups=3, padding=2)
            blurred_right= torch.nn.functional.conv2d(input[:, 3:6, :, :], self.blur_filter, groups=3, padding=2)
            gray_left    = (.299*input[:,0,:,:] + .587*input[:,1,:,:] + .114*input[:,2,:,:]).unsqueeze_(1)
            gray_right   = (.299*input[:,3,:,:] + .587*input[:,4,:,:] + .114*input[:,5,:,:]).unsqueeze_(1)

        gray_dsamp_left = nn.functional.conv2d(gray_left, self.dsamp_filter, stride=2)
        gray_dsamp_right= nn.functional.conv2d(gray_right, self.dsamp_filter, stride=2)
        dx_left         = nn.functional.conv2d(gray_dsamp_left, self.grad_filter)
        dy_left         = nn.functional.conv2d(gray_dsamp_left, self.grad_filter.transpose(-2,-1))
        dx_right        = nn.functional.conv2d(gray_dsamp_right, self.grad_filter)
        dy_right        = nn.functional.conv2d(gray_dsamp_right, self.grad_filter.transpose(-2,-1))
        gradient_left   = torch.cat([dx_left,dy_left], 1)
        gradient_right  = torch.cat([dx_right, dy_right], 1)

        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1_left = nn.parallel.data_parallel(self.model_rgb, blurred_left, self.gpu_ids)
            outs1_right= nn.parallel.data_parallel(self.model_rgb, blurred_right, self.gpu_ids)
            outs2_left = nn.parallel.data_parallel(self.model_gray, gray_left, self.gpu_ids)
            outs2_right= nn.parallel.data_parallel(self.model_gray, gray_right, self.gpu_ids)
            outs3_left = nn.parallel.data_parallel(self.model_grad, gradient_left, self.gpu_ids)
            outs3_right= nn.parallel.data_parallel(self.model_grad, gradient_right, self.gpu_ids)
        else:
            outs1_left = self.model_rgb(blurred_left)
            outs1_right= self.model_rgb(blurred_right)
            outs2_left = self.model_gray(gray_left)
            outs2_right= self.model_gray(gray_right)
            outs3_left = self.model_grad(gradient_left)
            outs3_right= self.model_grad(gradient_right)
        return outs1_left, outs1_right, outs2_left, outs2_right, outs3_left, outs3_right


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator_RGBonly(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, 
                 tensor=torch.FloatTensor, norm_layer=nn.BatchNorm2d, gpu_ids=[]):
        super(NLayerDiscriminator_RGBonly, self).__init__()
        self.gpu_ids     = gpu_ids
        self.grad_filter = tensor([0,0,0,-1,0,1,0,0,0]).view(1,1,3,3)
        self.input_nc    = input_nc
        self.model_rgb   = self.model(input_nc, ndf, n_layers, norm_layer)
        print('NLayerDiscriminator_RGBonly is selected!')

    def model(self, input_nc, ndf, n_layers, norm_layer):
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequences = [[
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.PReLU()
        ]]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequences += [[
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult + 1,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult + 1),
                nn.PReLU()
            ]]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequences += [[
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.PReLU(),
            \
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]]

        return SequentialOutput(*sequences)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            outs1_left = nn.parallel.data_parallel(self.model_rgb, 
                                                   input[:, 0:self.input_nc, :, :], 
                                                   self.gpu_ids)
            outs1_right= nn.parallel.data_parallel(self.model_rgb, 
                                                   input[:, self.input_nc:2*self.input_nc, :, :], 
                                                   self.gpu_ids)
        else:
            outs1_left = self.model_rgb(input[:, 0:self.input_nc, :, :])
            outs1_right= self.model_rgb(input[:, self.input_nc:2*self.input_nc, :, :])
        return outs1_left, outs1_right


class Plexer(nn.Module):
    def __init__(self):
        super(Plexer, self).__init__()

    def apply(self, func):
        for net in self.networks:
            net.apply(func)

    def cuda(self, device_id):
        for net in self.networks:
            net = nn.DataParallel(net)
            net.cuda(device_id)

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = [opt(net.parameters(), lr=lr, betas=betas) \
                           for net in self.networks]

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        self.optimizers[dom_b].step()

    def update_lr(self, new_lr):
        for opt in self.optimizers:
            for param_group in opt.param_groups:
                param_group['lr'] = new_lr

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save(net.cpu().state_dict(), filename)

    def load(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            net.load_state_dict(torch.load(filename))

class G_Plexer(Plexer):
    def __init__(self, n_domains, encoder, enc_args, decoder, dec_args,
                 block=None, shenc_args=None, shdec_args=None):
        super(G_Plexer, self).__init__()
        self.encoders = [encoder(*enc_args) for _ in range(n_domains)]
        self.decoders = [decoder(*dec_args) for _ in range(n_domains)]

        self.sharing = block is not None
        if self.sharing:
            self.shared_encoder = block(*shenc_args)
            self.shared_decoder = block(*shdec_args)
            self.encoders.append( self.shared_encoder )
            self.decoders.append( self.shared_decoder )
        self.networks = self.encoders + self.decoders

    def load_(self, save_dir, save_epoch):
        for i, net in enumerate(self.networks):
            if i == 0:
                net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G0.pth'))
            elif i==1:
                net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G1.pth'))
            elif i==2:
                net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G2.pth'))
            elif i==3:
                net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_G3.pth'))

    def init_optimizers(self, opt, lr, betas):
        self.optimizers = []
        for enc, dec in zip(self.encoders, self.decoders):
            params = itertools.chain(enc.parameters(), dec.parameters())
            self.optimizers.append( opt(params, lr=lr, betas=betas) )

    def net_in_trainmode(self, dom, cond):
        if cond:
            self.encoders[dom].train()
            self.decoders[dom].train()
        else:
            self.encoders[dom].eval()
            self.decoders[dom].eval()

    def dropout_in_train(self, dom, cond):
        def apply_dropout(m):
            if(type(m) == nn.Dropout):
                m.train()
        def remove_dropout(m):
            if(type(m) == nn.Dropout):
                m.eval()
        if cond:
            self.encoders[dom].apply(apply_dropout)
            self.decoders[dom].apply(apply_dropout)
        else:
            self.encoders[dom].apply(remove_dropout)
            self.decoders[dom].apply(remove_dropout)

    def forward(self, input, in_domain, out_domain):
        encoded = self.encode(input, in_domain)
        return self.decode(encoded, out_domain)

    def encode(self, input, domain):
        output = self.encoders[domain].forward(input)
        if self.sharing:
            return self.shared_encoder.forward(output, domain)
        return output

    def decode(self, input, domain):
        if self.sharing:
            input = self.shared_decoder.forward(input, domain)
        return self.decoders[domain].forward(input)

    def zero_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].zero_grad()
        if self.sharing:
            self.optimizers[-1].zero_grad()
        self.optimizers[dom_b].zero_grad()

    def step_grads(self, dom_a, dom_b):
        self.optimizers[dom_a].step()
        if self.sharing:
            self.optimizers[-1].step()
        self.optimizers[dom_b].step()

    def __repr__(self):
        e, d = self.encoders[0], self.decoders[0]
        e_params = sum([p.numel() for p in e.parameters()])
        d_params = sum([p.numel() for p in d.parameters()])
        return repr(e) +'\n'+ repr(d) +'\n'+ \
            'Created %d Encoder-Decoder pairs' % len(self.encoders) +'\n'+ \
            'Number of parameters per Encoder: %d' % e_params +'\n'+ \
            'Number of parameters per Deocder: %d' % d_params

class DispNet_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(DispNet_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def init_optimizers(self, opt, lr, betas):
        # Only create the optimizer for the dom_a (or nighttime dispNet)
        self.optimizers = [opt(self.networks[0].parameters(), lr=lr, betas=betas)]

    def net_in_trainmode(self, dom, cond):
        if cond:
            self.networks[dom].train()
        else:
            self.networks[dom].eval()

    def param_reguires_grad(self, dom, cond):
        for p in self.networks[dom].parameters():
            p.requires_grad = cond

    def forward(self, left, right, domain):
        dispnet = self.networks[domain]
        return dispnet.forward(left, right)

    def zero_grads(self, dom):
        self.optimizers[dom].zero_grad()

    def step_grads(self, dom):
        self.optimizers[dom].step()

    def load(self, save_path_0, save_path_1):
        for i, net in enumerate(self.networks):
            if i == 0:
                net.load_state_dict(torch.load(save_path_0)['state_dict'])
            elif i==1:
                net.load_state_dict(torch.load(save_path_1)['state_dict'])

    def save(self, save_path):
        for i, net in enumerate(self.networks):
            filename = save_path + ('%d.pth' % i)
            torch.save({'state_dict': net.state_dict()}, filename)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d DispNets' % len(self.networks) +'\n'+ \
            'Number of parameters per DispNet: %d' % t_params

class D_Plexer(Plexer):
    def __init__(self, n_domains, model, model_args):
        super(D_Plexer, self).__init__()
        self.networks = [model(*model_args) for _ in range(n_domains)]

    def load_(self, save_dir, save_epoch):
        for i, net in enumerate(self.networks):
            if i == 0:
                net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_D0.pth'))
            elif i==1:
                net.load_state_dict(torch.load(save_dir+'/'+str(save_epoch)+'_net_D1.pth'))

    def net_in_trainmode(self, dom, cond):
        if cond:
            self.networks[dom].train()
        else:
            self.networks[dom].eval()

    def forward(self, input, domain):
        discriminator = self.networks[domain]
        return discriminator.forward(input)

    def __repr__(self):
        t = self.networks[0]
        t_params = sum([p.numel() for p in t.parameters()])
        return repr(t) +'\n'+ \
            'Created %d Discriminators' % len(self.networks) +'\n'+ \
            'Number of parameters per Discriminator: %d' % t_params


class SequentialContext(nn.Sequential):
    def __init__(self, n_classes, *args):
        super(SequentialContext, self).__init__(*args)
        self.n_classes = n_classes
        self.context_var = None

    def prepare_context(self, input, domain):
        if self.context_var is None or self.context_var.size()[-2:] != input.size()[-2:]:
            tensor = torch.cuda.FloatTensor if isinstance(input.data, torch.cuda.FloatTensor) \
                     else torch.FloatTensor
            self.context_var = tensor(*((1, self.n_classes) + input.size()[-2:]))

        self.context_var.data.fill_(-1.0)
        self.context_var.data[:,domain,:,:] = 1.0
        return self.context_var

    def forward(self, *input):
        if self.n_classes < 2 or len(input) < 2:
            return super(SequentialContext, self).forward(input[0])
        x, domain = input

        for module in self._modules.values():
            if 'Conv' in module.__class__.__name__:
                context_var = self.prepare_context(x, domain)
                x = torch.cat([x, context_var], dim=1)
            elif 'Block' in module.__class__.__name__:
                x = (x,) + input[1:]
            x = module(x)
        return x

class SequentialOutput(nn.Sequential):
    def __init__(self, *args):
        args = [nn.Sequential(*arg) for arg in args]
        super(SequentialOutput, self).__init__(*args)

    def forward(self, input):
        predictions = []
        layers = self._modules.values()
        for i, module in enumerate(layers):
            output = module(input)
            if i == 0:
                input = output;  continue
            predictions.append( output[:,-1,:,:] )
            if i != len(layers) - 1:
                input = output[:,:-1,:,:]
        return predictions
 
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

class PerceptualLossVgg16(nn.Module):
    def __init__(self, vgg=None, gpu_ids=[0], weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg16, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16().cuda()
        else:
            self.vgg = vgg
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
            # print(len(x_vgg), x_vgg[0].size(), y_vgg[0].size())
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

# From https://github.com/Vandermode/ERRNet/blob/master/models/vgg.py
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        self.vgg_pretrained_features = visionmodels.vgg19(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [2, 7, 12, 21, 30] # assuming 1 starting index!
        out = []
        #indices = sorted(indices)
        for i in range(indices[-1]):
            X = self.vgg_pretrained_features[i](X)
            if (i+1) in indices:
                out.append(X)
        return out

# From https://github.com/Vandermode/ERRNet/blob/c7d786f4d94e8ac1f56fa3de75b53cbe9079e527/models/losses.py#L111
class PerceptualLossVgg19(nn.Module):
    def __init__(self, vgg=None, gpu_ids=[0], weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg19, self).__init__()        
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.vgg = nn.DataParallel(self.vgg, device_ids=gpu_ids)
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0/2.6, 1.0/4.8, 1.0/3.7, 1.0/5.6, 10/1.5]
        self.indices = indices or [2, 7, 12, 21, 30]
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

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class Vgg16RotNet(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16RotNet, self).__init__()
        num_classes = 4
        # Features
        relu1_2 = visionmodels.vgg16(pretrained=False).features[:4]
        relu2_2 = visionmodels.vgg16(pretrained=False).features[4:9]
        relu3_3 = visionmodels.vgg16(pretrained=False).features[9:16]
        relu4_3 = visionmodels.vgg16(pretrained=False).features[16:23]
        pool5_1 = visionmodels.vgg16(pretrained=False).features[23:31]
        # AvgPool
        avgpool = nn.AdaptiveAvgPool2d((7,7))
        # Classifier
        fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(25088, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )
        classifier = nn.Sequential(
            nn.Linear(4096, num_classes),
        )
        self._feature_blocks = nn.ModuleList([
            relu1_2,
            relu2_2,
            relu3_3,
            relu4_3,
            pool5_1,
            fc_block,
            classifier,
        ])
        self.all_feat_names = [
            'relu1_2',
            'relu2_2',
            'relu3_3',
            'relu4_3',
            'pool5_1',
            'fc_block',
            'classifier',
        ]
        assert(len(self.all_feat_names) == len(self._feature_blocks))
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                # print(param, param.requires_grad)

    def forward(self, X, indices):
        # relu1_2
        if indices[0] == 3: 
            feat_block_id = 0
        # relu2_2
        elif indices[0] == 8:
            feat_block_id = 1
        # relu3_3
        elif indices[0] == 15:
            feat_block_id = 2
        # relu4_3
        elif indices[0] == 22:
            feat_block_id = 3
        else:
            print('Index %d not supported at the moment!' % indices[0])
        preds = []
        for f in range(feat_block_id+1):
            X = self._feature_blocks[f](X)
        preds.append(X)
        return preds

class PerceptualLossVgg16RotNet(nn.Module):
    def __init__(self, vgg=None, load_model=None, gpu_ids=[0], weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg16RotNet, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16RotNet().cuda()
        else:
            self.vgg = vgg
        self.vgg = nn.DataParallel(self.vgg, device_ids=gpu_ids)
        if load_model is None:
            print('PerceptualLossVgg16RotNet needs a pre-trained checkpoint!')
            raise Exception
        else:
            print('Vgg16RotNet initialized with %s'% load_model)
            model_state_dict = torch.load(load_model)
            self.vgg.load_state_dict(model_state_dict['network'])
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0]
        self.indices = indices or [22] # assuming 1 starting index!
        assert(len(self.indices)==1)   # More than 1 index not supported at the momemt!
        assert(len(self.weights)==1)   # More than 1 index not supported at the momemt!
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
            # print(len(x_vgg), x_vgg[0].size(), y_vgg[0].size())
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Vgg16ExDark(torch.nn.Module):
    def __init__(self, load_model=None, requires_grad=False):
        super(Vgg16ExDark, self).__init__()
        # Create the model
        self.vgg_pretrained_features = visionmodels.vgg16(pretrained=True).features
        if load_model is None:
            print('Vgg16ExDark needs a pre-trained checkpoint!')
            raise Exception
        else:
            print('Vgg16ExDark initialized with %s'% load_model)
            model_state_dict = torch.load(load_model)
            model_dict       = self.vgg_pretrained_features.state_dict()
            # The checkpoint has keys wuth 'module.features.',
            # for k, v in model_state_dict.items():
                # print(k[16:])
            model_state_dict = {k[16:]: v for k, v in model_state_dict.items() if k[16:] in model_dict}
            self.vgg_pretrained_features.load_state_dict(model_state_dict)
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

"""
Tutorials/Helpers
https://gist.github.com/brucemuller/37906a86526f53ec7f50af4e77d025c9
https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
"""
class PerceptualLossVgg16ExDark(nn.Module):
    def __init__(self, vgg=None, load_model=None, gpu_ids=[0], weights=None, indices=None, normalize=True):
        super(PerceptualLossVgg16ExDark, self).__init__()        
        if vgg is None:
            self.vgg = Vgg16ExDark(load_model)
        else:
            self.vgg = vgg
        self.vgg = nn.DataParallel(self.vgg, device_ids=gpu_ids).cuda()
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
            # print(len(x_vgg), x_vgg[0].size(), y_vgg[0].size())
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss
