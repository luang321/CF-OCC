
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import MultiStepLR
import torchvision.ops as ops
from functools import reduce
from einops import rearrange
from einops.layers.torch import Rearrange
import time
from easydict import EasyDict as edict
import yaml
import copy
import datetime
import os
from .mmdet_wrapper import MMDetWrapper
from .unet3d import UNet3D


import random

def random_select(input_list):
    if not input_list:
        return None  # 如果列表为空，返回None
    return random.choice(input_list)


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def generate_meshgrid(features):
    b, c, h, w = features.shape
    v_coords, u_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    u_coords = u_coords.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    v_coords = v_coords.unsqueeze(0).unsqueeze(0).expand(b, 1, h, w)
    meshgrid = torch.cat((u_coords, v_coords), dim=1)
    
    return meshgrid

def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)

def pix2cam(p_pix, depth, K):
    p_pix = torch.cat([p_pix * depth, depth], dim=1)  # bs, 3, h, w
    return K.inverse() @ p_pix.flatten(2)

def cam2vox(p_cam, E, vox_origin, vox_size, offset=0.5):
    p_wld = E.inverse() @ F.pad(p_cam, (0, 0, 0, 1), value=1)
    p_vox = (p_wld[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(0).unsqueeze(0)) / vox_size - offset
    return p_vox

def pix2vox(p_pix, depth, K, E, vox_origin, vox_size, offset=0.5, downsample_z=1):
    p_cam = pix2cam(p_pix, depth, K)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size, offset)
    if downsample_z != 1:
        p_vox[..., -1] /= downsample_z
    return p_vox
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, bn_momentum = 0.1):
        super(Upsample, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                dilation=1,
                output_padding=1,
            ),
        )
        self.norm = nn.BatchNorm3d(out_channels, momentum=bn_momentum)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.main(x)
        x = self.relu(self.norm(x))
        return x

class tempnet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        bn_momentum = 0.1
        self.dim_2d = config.dim_2d #config.dim_2d
        dim = config.dim_3d
        self.fold_size = config.fold_2d
        self.future_length = config.future_length #5
        self.past_length = config.past_length
        self.thr = config.thr

        self.stride =  config.stride
        self.freeze_backbone  = config.freeze_backbone
        self.build_rgb_net(config.net_rgb)

        d = dim //2
        s = self.stride//4
        self.n_bins = 128
        self.unet3d = UNet3D(dim)

        # UNet3D(dim)
        self.head =  nn.Sequential(Upsample(dim, d),Upsample(d, d),nn.Conv3d(d, d, kernel_size=1),nn.ReLU(),nn.Conv3d(d, int(config.n_classes*(s**3)), kernel_size=1),
                            Rearrange("b (p1 p2 p3 c) h w d -> b c (h p1) (w p2) (d p3)",p1=s,p2=s,p3=s))
        #
        input_dim =   int(self.dim_2d*(self.future_length + self.past_length + 1))#
        self.fc = nn.Sequential(
            nn.Linear(input_dim, int(dim*2)),
            nn.LayerNorm(int(dim*2)),
            nn.ReLU(),
            nn.Linear(int(dim*2), dim))

        self.feat_shape = [int(i//self.stride) for i in config.full_scene_size]
        self.scene_shape = config.full_scene_size
        print(self.scene_shape,self.stride)
        #self.fill = nn.Parameter(torch.randn(1,dim,*self.feat_shape))
    def build_rgb_net(self,name):
        self.project_res = [4, 8, 16,32]
        num_input_images = 1  #self.future_length + 1
        self.net_rgb = MMDetWrapper(
            checkpoint_path =  os.path.join(os.getcwd(),"backbone_pth","maskdino_r50_50e_300q_panoptic_pq53.0.pth"),
                freeze= self.freeze_backbone,scales = self.project_res,num_input_images = num_input_images  ) 
            # os.path.join(os.getcwd(),"backbone_pth","maskdino_r50_50e_300q_panoptic_pq53.0.pth")
        self.out_projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.net_rgb.hidden_dims, int(self.dim_2d * num_input_images), 1),
                nn.BatchNorm2d(int(self.dim_2d * num_input_images)),
                nn.ReLU(inplace=True),
            ) for _ in self.project_res
        ])

    def forward_2d(self,img_2,freeze = False):
        if self.freeze_backbone or freeze:
            with torch.no_grad():
                x_rgb = self.net_rgb(img_2)
        else:
            x_rgb = self.net_rgb(img_2)
        fs = list()
        for ii, s in enumerate(self.project_res):
            f = self.out_projs[ii](x_rgb["1_" + str(s)].contiguous()).contiguous()
            if s != self.fold_size:
                f = F.interpolate(f,scale_factor=s/self.fold_size, mode='bilinear')
            fs.append(f)
        fs = sum(fs)
        return fs
    
    def sample(self,imgs,frontiers,sample_points,sample_depths = None):
        bs = imgs.shape[0]
        assert bs == 1
        frontiers = [f.float().mean(1) for f in frontiers]
        valids = [f>self.thr for f in frontiers]
        candidate = [i for i,f in enumerate(valids) if f.sum()>100]
        candidate = random_select(candidate)
        frontiers_all = torch.stack(valids).any(0)
        idxs = torch.nonzero(frontiers_all, as_tuple=False)

        srcs = torch.zeros(len(idxs),len(frontiers),self.dim_2d).to(imgs)
        depth_weights = torch.zeros(len(idxs),len(frontiers)).to(imgs)
        weights = torch.zeros(len(idxs),len(frontiers)).to(imgs) #torch.ones(len(idxs)).to(imgs)#
        for i,(samplt_pt,frontier,valid) in enumerate(zip(sample_points,frontiers,valids)):
            feat = self.forward_2d(imgs[:,i],freeze = (i!= candidate))

            idx_i = torch.nonzero(valid , as_tuple=False)
            samplt_pt = samplt_pt[idx_i[:,0],:,idx_i[:,1],idx_i[:,2],idx_i[:,3]][None,None]

            src = F.grid_sample(feat, samplt_pt, mode='bilinear', align_corners=True,padding_mode = 'zeros')
            src = rearrange(src,"b c h w -> b (h w) c").contiguous()
            indice = torch.nonzero(valid[idxs[:,0],idxs[:,1],idxs[:,2],idxs[:,3]], as_tuple=False)[:,0]
            #print(i)
            if sample_depths is not None:
                depth = sample_depths[i]
                depth = 51.2 -depth[idx_i[:,0],0,idx_i[:,1],idx_i[:,2],idx_i[:,3]]
                depth_weights[indice,i] = depth
            srcs[indice,i] = src[0]
            weights[indice,i] = frontier[idx_i[:,0],idx_i[:,1],idx_i[:,2],idx_i[:,3]]
        #print(depth_weights.max(1,keepdim = True)[0],depth_weights.max(1,keepdim = True)[0].shape,weights.shape)
        srcs = srcs.flatten(1)
        #print(srcs.shape)
        srcs = self.fc(srcs[None])
        return srcs,frontiers_all

    
    
    def restore(self,src,frontier):
        _,_,c = src.shape
        b,h,w,d = frontier.shape
        assert b == 1
        new_src = torch.zeros(b,c,h,w,d).to(src)
        idxs = torch.nonzero(frontier, as_tuple=False)
        new_src[idxs[:,0],:,idxs[:,1],idxs[:,2],idxs[:,3]] = src[0]
        #new_src = new_src + self.fill
        #print(self.thr)
        #print(self.fill.shape,new_src.shape)
        return new_src
    


    def forward(self,batch,frontiers,sample_points, sample_depths = None,psuedo_imgs = None):
        #,batch,frontiers,sample_points,sample_depths = None
        imgs= batch["img"] if psuedo_imgs is None else psuedo_imgs
        b = imgs.shape[0]
        src,frontiers_all = self.sample(imgs,frontiers,sample_points,sample_depths =sample_depths )
        src = self.restore(src,frontiers_all)
        ssc_pred = self.head(self.unet3d(src))

        cond = dict()
        cond["ssc_pred"] = ssc_pred
        return cond

    def cal_feature_loss(self,img1,img2,loss = 0,loss_dict = dict(),aux = "" ):
        pred =  self.net_rgb(img1)
        with torch.no_grad():
            target = self.net_rgb(img2)
        for ii,(k,t) in enumerate(target.items()):
            l1_loss = (pred[k] - t).abs().mean()
            loss += l1_loss
            loss_dict["feat_loss" + k + aux] = l1_loss.detach()
        return loss,loss_dict

    @classmethod
    def init_from_path(cls_ ,path):
        from misc import init_dataset_config
        with open(os.path.join(path,"conf.yaml"), "r") as f:
            config = edict(yaml.safe_load(f)) 
            config = init_dataset_config(config)
        generator  = cls_(config)
        weight  = torch.load(os.path.join(path,"checkpoints","last.ckpt"), map_location='cpu')["model_g"]
        weight = {k[8:]:v for k,v in weight.items() if "tempnet" in k}

        mis,uxp = generator.load_state_dict(weight,strict = False)
        print("load form:",path ,mis,uxp)
        return generator

def convbn_2d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=pad, bias=False),
        nn.BatchNorm2d( out_channels)
    )


class SimpleUnet(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(SimpleUnet, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d(in_channels, in_channels * 2, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 2, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 4, 3, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            convbn_2d(in_channels * 4, in_channels * 4, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels * 2)
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.redir1 = convbn_2d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_2d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)
        return self.out(conv6) 