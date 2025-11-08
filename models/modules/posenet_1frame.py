
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import torchvision.ops as ops
from einops import rearrange
from einops.layers.torch import Rearrange

import copy
import os
from .mmdet_wrapper import MMDetWrapper
from .unet3d import Down_layer


import random
class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

def random_select(input_list):
    if not input_list:
        return None  # 如果列表为空，返回None
    return random.choice(input_list)
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class FinalLayer(nn.Module):
    def __init__(self, hidden_size, dim,  n_frames):
        super().__init__()
        self.n_frames = n_frames
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.linear_rot = nn.Linear(
            hidden_size,int(n_frames*3), bias=True
        )
        self.linear_trans = nn.Linear(
            hidden_size,int(n_frames*3), bias=True
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(min(dim, 1024), 2 * hidden_size, bias=True),
        )


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        rot = self.linear_rot(x).view(x.shape[0],self.n_frames,-1)
        trans = self.linear_trans(x).view(x.shape[0],self.n_frames,-1)
        return rot,trans
class posenet(nn.Module):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.future_length = config.future_length #5
        self.past_length = config.past_length
        self.thr = config.thr

        bn_momentum = 0.1
        self.dim_2d = config.dim_posenet #config.dim_2d
        dim = self.dim_2d
        self.fold_size = config.fold_2d
        self.stride =  config.stride
        self.freeze_backbone  = config.freeze_backbone
        self.build_rgb_net(config.net_rgb)
        self.unet3d = Down_layer(dim,affine = False)
        self.t_embbeder = MLP(int((self.past_length+1)*16),dim,dim,3)
        self.head = FinalLayer(dim,dim,self.future_length)

        input_dim =  self.dim_2d# self.dim_2d #int(self.dim_2d*(self.future_length + 1))
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


    def sample(self,imgs,frontiers,sample_points):
        bs = imgs.shape[0]
        assert bs == 1
        valid = frontiers.float().mean(1)>self.thr 
        #print(valid.shape,frontiers.shape)
        idxs = torch.nonzero(valid , as_tuple=False)
        samplt_pt = sample_points[idxs[:,0],:,idxs[:,1],idxs[:,2],idxs[:,3]][None,None]
        feat = self.forward_2d(imgs,freeze =False)
        src = F.grid_sample(feat, samplt_pt, mode='bilinear', align_corners=True,padding_mode = 'zeros')
        src = rearrange(src,"b c h w -> b (h w) c").contiguous()
        #print(src.shape)
        srcs = self.fc(src)
        return srcs,valid
    def restore(self,src,frontier):
        _,_,c = src.shape
        b,h,w,d = frontier.shape
        assert b == 1
        new_src = torch.zeros(b,c,h,w,d).to(src)
        idxs = torch.nonzero(frontier, as_tuple=False)
        new_src[idxs[:,0],:,idxs[:,1],idxs[:,2],idxs[:,3]] = src[0]
        return new_src
    def forward(self,batch,frontiers,sample_points,known_poses):
        imgs = batch["img"][:,self.past_length]
        frontiers,sample_points = frontiers[self.past_length],sample_points[self.past_length]
        t = torch.cat(known_poses,-1).flatten(1)
        t = self.t_embbeder(t.float())

        b = imgs.shape[0]
        src,frontiers_all = self.sample(imgs,frontiers,sample_points)
        src = self.restore(src,frontiers_all)
        src = self.unet3d(src,t)

        src = src.flatten(2).max(2)[0]
        axis_angle,trans = self.head(src,t)

        outputs = dict()
        outputs["axis_angle"] = axis_angle
        outputs["trans"] = trans
        return outputs



    @classmethod
    def init_from_path(cls_ ,path):
        from misc import init_dataset_config
        with open(os.path.join(path,"conf.yaml"), "r") as f:
            config = edict(yaml.safe_load(f)) 
            config = init_dataset_config(config)
        generator  = cls_(config)
        weight  = torch.load(os.path.join(path,"checkpoints","last.ckpt"), map_location='cpu')["model"]
        weight = {k[8:]:v for k,v in weight.items() if "tempnet" in k}
        mis,uxp = generator.load_state_dict(weight,strict = False)
        print("load form:",path ,mis,uxp)
        return generator
