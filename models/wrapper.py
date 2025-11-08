import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributed import all_reduce
from torch import nn
import numpy as np
import math
from inspect import isfunction
from misc import get_obj_from_str,all_gather,is_main_process,mkdir_if_missing
from torch.optim.lr_scheduler import MultiStepLR

import copy
from easydict import EasyDict as edict
import yaml
import os 
from einops import rearrange,repeat
from models.loss.sscMetrics import SSCMetrics
from models.loss.ssc_loss import sem_scal_loss, CE_ssc_loss,  geo_scal_loss
#from layers.Voxel_Level.gen_denoise import Denoise
#from utils.loss import *
from torchvision.utils import save_image
"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

def denormalize_and_save(tensor,save_path = None, mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]):
    """[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    对归一化的张量进行反归一化操作
    
    Args:
        tensor (torch.Tensor): 归一化的张量，形状为 (C, H, W) 或 (B, C, H, W)
        mean (list or tuple): 各通道的均值，例如 [0.485, 0.456, 0.406]（ImageNet均值）
        std (list or tuple): 各通道的标准差，例如 [0.229, 0.224, 0.225]（ImageNet标准差）
    
    Returns:
        torch.Tensor: 反归一化后的张量
    """
    if not isinstance(mean, torch.Tensor):
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor)  # (C, 1, 1)
    if not isinstance(std, torch.Tensor):
        std = torch.tensor(std).view(-1, 1, 1).to(tensor)   # (C, 1, 1)
    
    # 反归一化：tensor * std + mean
    denorm_tensor = tensor * std + mean
    
    # 确保值在 [0, 1] 范围内（如果输入是归一化的图像）
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    if save_path is not None and is_main_process():
        print(save_path)
        save_image(denorm_tensor,save_path)
    return denorm_tensor

def sum_except_batch(x, num_dims=1,mask = None):
    if mask is not None:
        x = x*mask.float()
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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
def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

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


class wrapper(torch.nn.Module):
    def __init__(self, config):
        super(wrapper, self).__init__()
        self.lr =config.lr
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        self.num_classes = config.n_classes
        self.class_names = config.class_names
        self.class_weights = config.class_weights

        self.future_length = config.future_length
        self.past_length = config.past_length
        self.mode = config.mode if "mode" in config else  "tempnet"
        if self.mode == "tempnet":
            self.tempnet = get_obj_from_str(config.tempnet)(config) #train spatio temperal ssc
        else:
            #train pose/synth net
            self.tempnet = get_obj_from_str(config.tempnet).init_from_path(config.tempnet_pth)
            toggle_grad(self.tempnet,False)

            if "psuedo_pth" in config:
                self.synthnet = get_obj_from_str(config.synthnet).init_from_path(config.synthnet_pth)
                toggle_grad(self.synthnet,False)
                self.train_psuedo = False
            else:
                self.synthnet = get_obj_from_str(config.synthnet)(config)
                self.train_psuedo = True

            if "posenet_pth" in config:
                self.posenet  = get_obj_from_str(config.posenet).init_from_path(config.posenet_pth)
                toggle_grad(self.posenet,False)
                self.train_pose = False
            else:
                self.posenet  = get_obj_from_str(config.posenet)(config)
                self.train_pose = True

            
        self.eps = config.eps if "eps" in config else 1e-30
        self.metrics = {"val": SSCMetrics(self.num_classes), "val_n": SSCMetrics(self.num_classes)
                        , "val_m": SSCMetrics(self.num_classes), "val_f":SSCMetrics(self.num_classes)}

        self.ssim = SSIM()
        self.stride = config.stride
        self.thr = config.thr
        self.scene_shape = config.full_scene_size

        image_shape = [384,1280]
        image_grid = generate_grid(image_shape)
        image_grid = torch.flip(image_grid, dims=[0]).unsqueeze(0)  # 2(wh), h, w
        self.register_buffer('image_grid', image_grid.float())
        print("num_cls:",self.num_classes,self.eps)

    def device(self):
        return self.denoise_fn.device
    def final_loss(self,x,target,aux = "",loss = 0,loss_dict = {},b = 0):
        class_weight = self.class_weights.type_as(x)
        loss_ssc = CE_ssc_loss(x , target, class_weight,b = b)
        #print(b)
        loss += loss_ssc
        loss_dict["loss_ssc" + aux] = loss_ssc.detach()
    
        loss_sem_scal = sem_scal_loss(x, target,b = b)
        loss += loss_sem_scal
        loss_dict[ "loss_sem_scal"+ aux] = loss_sem_scal.detach()
        
        loss_geo_scal = geo_scal_loss(x, target,b = b)
        loss += loss_geo_scal
        loss_dict[ "loss_geo_scal"+ aux] = loss_geo_scal.detach()
        return loss,loss_dict

    def project(self,lidar_pts,rela_pose,K,E):
        #b, n, _ = lidar_pts.shape
        #lidar_pts = torch.cat([lidar_pts, torch.ones(b, n, 1, device=lidar_pts.device)], dim=-1)
        b, n, _ = lidar_pts.shape
        lidar_pts = torch.cat([lidar_pts, torch.ones(b, n, 1, device=lidar_pts.device)], dim=-1)
        cam_pts =  torch.bmm(lidar_pts, E.transpose(1, 2))
        cam_pts = torch.bmm(cam_pts,rela_pose.transpose(1, 2))
        #canon_lidar_pts = lidar_pts
        
        projected_pix = torch.bmm(cam_pts[...,:3], K.transpose(1, 2))
        projected_pix[..., :2] = projected_pix[..., :2] / projected_pix[..., 2:3]
        return projected_pix

    def get_frontier_1_frame(self,depth,sample_point):
        
        sample_point = copy.deepcopy(sample_point)
        sample_point,sample_d = sample_point[...,:2],sample_point[...,2:3]
        sample_point[...,0] = sample_point[...,0]/depth.shape[-1]
        sample_point[...,1] = sample_point[...,1]/depth.shape[-2]
        valid_mask = ((sample_point>0)&(sample_point<1)).all(-1)
        sample_point = sample_point.clamp(0,1)*2-1
        dep = F.grid_sample(depth, sample_point.unsqueeze(1), mode="nearest", align_corners=True,padding_mode = 'zeros') #'bilinear'
        dep = rearrange(dep,"b c k n -> b n (c k)").contiguous()
        frontier = ((dep - sample_d)[...,0].abs() <0.5) & valid_mask & (sample_d[...,0]>=0) & (sample_d[...,0]<51.2)
        
        # & (sample_d[...,0]<25.6)
        #((dep - sample_d)[...,0].abs() <2.5) & valid_mask & (sample_d[...,0]>=0) & (sample_d[...,0]<51.2)

        w,h,d = self.scene_shape
        frontier = rearrange(frontier,"b (h w d) -> b h w d",h = h,w = w,d = d)
        frontier_down = rearrange(frontier,"b (h p1) (w p2) (d p3) -> b (p1 p2 p3) h w d",p1 = self.stride,p2 = self.stride,p3 = self.stride).contiguous()
        
        sample_point = rearrange(copy.deepcopy(sample_point),"b (h w d) c -> b c h w d",h = h,w = w,d = d)
        sample_point = rearrange(sample_point,"b c (h p1) (w p2) (d p3) -> b (p1 p2 p3) c h w d",p1 = self.stride,p2 = self.stride,p3 = self.stride).contiguous().mean(1)
        #print(sample_point.max(),sample_point.min())
        sample_d = rearrange(copy.deepcopy(sample_d),"b (h w d) c -> b c h w d",h = h,w = w,d = d)
        sample_d = rearrange(sample_d,"b c (h p1) (w p2) (d p3) -> b (p1 p2 p3) c h w d",p1 = self.stride,p2 = self.stride,p3 = self.stride).contiguous().mean(1)
        
        return frontier,frontier_down,sample_point,sample_d

    def get_frontier(self,batch, pred_poses = None,psuedo_depth = None):
        lidar_pts,K,E = batch["lidar_pts"][0].float(),batch["K"].float(),batch["E"].float()
        depth =  batch["depth"] if psuedo_depth is None else psuedo_depth
        relative_poses = batch["relative_poses"] if pred_poses is None else pred_poses
        frontiers = list()
        frontiers_down = list()
        sample_points = list()
        sample_depths = list()
        n_frames = depth.shape[1]
        #print(depth.shape)
        for i in range(n_frames):
            sample_point = self.project(copy.deepcopy(lidar_pts),relative_poses[i].float(),K,E)
            frontier,frontier_down,sample_point,sample_d = self.get_frontier_1_frame(depth[:,i:i+1],sample_point)
            frontiers.append(frontier)
            frontiers_down.append(frontier_down)
            sample_points.append(sample_point)
            sample_depths.append(sample_d)
        return frontiers,frontiers_down,sample_points,sample_depths
    def forward_tempnet(self,batch,loss = 0, loss_dict = dict(),step_type = "train"):
        vis,cond = {},{}
        y_true  = batch["voxel_label"].long()
        frontiers,frontiers_down,sample_points,sample_depths = self.get_frontier(batch)
        outputs = self.tempnet(batch,frontiers_down,sample_points,sample_depths)
        ssc_pred = outputs["ssc_pred"]
        loss,loss_dict = self.final_loss(ssc_pred,y_true,loss= loss, loss_dict = loss_dict)

        if "depth_volume" in outputs :
            depth_volume = outputs["depth_volume"]
            gt_depth = self.get_gt_depth(batch)
            depth_loss = self.get_klv_depth_loss(gt_depth,depth_volume)
            depth_loss = depth_loss *0.001
            loss += depth_loss
            loss_dict["depth_loss"] = depth_loss.detach()
        y_pred = ssc_pred.detach().softmax(1).argmax(1)
        
        for i,m in enumerate(frontiers):
            y_vis = copy.deepcopy(y_true)
            y_vis[~m] = 0
            vis[f"y_true_{i}"] = y_vis
        vis["y_true"] = y_true
        vis["y_pred"] = y_pred
        cond["vis"] = vis
        if step_type == "val":
            self.update_metric(y_pred,y_true)
        return loss,loss_dict,cond
    
    def warp_once(self,img,depth,pose,K,E):
        '''
        cam_pts = pix2cam(self.image_grid, depth, K)
        cam_pts = torch.bmm(pose,cam_pts)
        projected_pix = torch.bmm(K,cam_pts[...,:3])
        projected_pix[:, :2] = projected_pix[:, :2] / projected_pix[:, 2:3]
        '''
        cam_pts = pix2cam(self.image_grid, depth, K).transpose(1, 2)

        b, n, _ = cam_pts.shape
        cam_pts  = torch.cat([cam_pts , torch.ones(b, n, 1, device=cam_pts.device)], dim=-1)
        cam_pts = torch.bmm(cam_pts,pose.transpose(1, 2).float())
        projected_pix = torch.bmm(cam_pts[...,:3], K.transpose(1, 2))
        projected_pix[..., :2] = projected_pix[..., :2] / projected_pix[..., 2:3]
        projected_depth = projected_pix[..., 2:3]
        projected_pix = projected_pix[..., :2].round()

        valid_mask = (projected_pix[...,0]>=0)&(projected_pix[...,0]<1280)&(projected_pix[...,1]>=0)&(projected_pix[...,1]<384)
        valid_idx = torch.nonzero(valid_mask,as_tuple = False)
        valid_pix = projected_pix[valid_mask,:].long()
        projected_depth = projected_depth[valid_mask,:].float()
        
        normed_sample_pts = copy.deepcopy(self.image_grid).flatten(2).transpose(1, 2)[valid_mask,:]#copy.deepcopy(valid_pix).float()
        normed_sample_pts[...,0] = normed_sample_pts[...,0]/1280
        normed_sample_pts[...,1] = normed_sample_pts[...,1]/384
        normed_sample_pts = normed_sample_pts.clamp(0,1)*2 -1

        warped_img = torch.zeros_like(img)
        warped_depth = torch.ones_like(self.image_grid[:,:1])*10
        sampled_img = F.grid_sample(input=img, grid=normed_sample_pts[None,None], mode='bilinear', padding_mode='border',
                                           align_corners=True)[0,:,0]
        #print(normed_sample_pts.shape,sampled_img.shape,projected_depth.shape,valid_pix.shape,warped_img.shape,warped_img[valid_idx[...,0],:,valid_pix[...,1],valid_pix[...,0]].shape)
        warped_img[valid_idx[...,0],:,valid_pix[...,1],valid_pix[...,0]] = sampled_img.permute(1,0) 
        warped_depth[valid_idx[...,0],:,valid_pix[...,1],valid_pix[...,0]] = projected_depth
        '''
        target_pix = copy.deepcopy(self.image_grid).flatten(2).transpose(1, 2)[valid_mask,:]
        sample_pts = rearrange(copy.deepcopy(self.image_grid),"b c h w -> b h w c")
        sample_pts[valid_idx[...,0],valid_pix[...,1],valid_pix[...,0],:] = target_pix

        normed_sample_pts = copy.deepcopy(sample_pts)
        normed_sample_pts[...,0] = normed_sample_pts[...,0]/1280
        normed_sample_pts[...,1] = normed_sample_pts[...,1]/384
        normed_sample_pts = normed_sample_pts.clamp(0,1)*2 -1

        warped_img = F.grid_sample(input=img, grid=normed_sample_pts, mode='bilinear', padding_mode='border',
                                           align_corners=True)
        warped_depth = F.grid_sample(input=depth, grid=normed_sample_pts, mode='bilinear', padding_mode='border',
                                           align_corners=True)'''

        return warped_img,warped_depth
    def warp(self,known_imgs,known_depths,knwon_poses,K,E):
        warped_inputs = list()
        for i,pose in enumerate(knwon_poses):
            img,depth = known_imgs[:,i],known_depths[:,i:i+1]
            warped_img,warped_depth = self.warp_once(img,depth,pose,K,E)
            warped_input = torch.cat((warped_img,warped_depth),1)
            warped_inputs.append(warped_input)
            #denormalize_and_save(warped_img[0],os.path.join("/data/lha/itsc2025_ssc/vis/psuedo",f"warped_{i}.png"))
            #denormalize_and_save(warped_img[0],os.path.join("/data/lha/itsc2025_ssc/vis/warp",f"fake_{i}.png"))
            #denormalize_and_save(img[0],os.path.join("/data/lha/itsc2025_ssc/vis/warp",f"real_{i}.png"))
        warped_inputs = torch.cat(warped_inputs,1)
        return warped_inputs

    def forward_synth_1_frame(self,warped_input,target_img,target_depth,loss= 0 ,loss_dict ={},aux = ""):
        outputs = self.synthnet(warped_input)
        psuedo_img,psuedo_depth = outputs["psuedo_img"],outputs["psuedo_depth"]
        img_loss_ssim = self.ssim(psuedo_img, target_img).mean()
        img_loss_l1 =  (psuedo_img - target_img).abs().mean()
        depth_loss = (psuedo_depth - target_depth).abs().mean()
        #denormalize_and_save(target_img[0],os.path.join("/data/lha/itsc2025_ssc/vis/warp","gt.png"))
        
        loss =  loss + img_loss_ssim + img_loss_l1 + depth_loss
        loss_dict["img_loss_ssim" + aux] = img_loss_ssim
        loss_dict["img_loss_l1" + aux] = img_loss_l1
        loss_dict["depth_loss" + aux] = depth_loss

        loss,loss_dict = self.tempnet.cal_feature_loss(psuedo_img,target_img,loss = loss,loss_dict = loss_dict,aux = aux)
        return loss,loss_dict,psuedo_img.detach(),psuedo_depth.detach()

    def forward_synth(self,batch,loss = 0, loss_dict = dict(),step_type = "train"):
        imgs,depths = batch["img"],batch["depth"]
        poses =  batch["relative_poses"] if step_type == "train" else batch["pred_poses"]
        K,E = batch["K"].float(),batch["E"].float()
        for i in range(self.future_length): 
            known_imgs,known_depths,knwon_poses = imgs[:,i:self.past_length+1 +i],depths[:,i:self.past_length+1 +i],poses[i:self.past_length+1 +i]
            target_img,target_depth,canon_pose = imgs[:,self.past_length+1 +i],depths[:,self.past_length+1 +i],poses[self.past_length+1 +i]
            knwon_poses = [torch.matmul(canon_pose,p.inverse()) for p in knwon_poses]
            warped_input = self.warp(known_imgs,known_depths,knwon_poses,K,E)
            loss,loss_dict,psuedo_img,psuedo_depth = self.forward_synth_1_frame(warped_input,target_img,target_depth[:,None],loss= loss ,loss_dict =loss_dict,aux = f"_{i}")
            if step_type != "train":
                #denormalize_and_save(warped_input[0][-4:-1],os.path.join("/data/lha/itsc2025_ssc/vis",floder,f"warped_{i}.png"))
                #denormalize_and_save(psuedo_img[0],os.path.join("/data/lha/itsc2025_ssc/vis",floder,f"psuedo_{i}.png"))
                #denormalize_and_save(imgs[:,self.past_length+1 +i][0],os.path.join("/data/lha/itsc2025_ssc/vis",floder,f"gt_{i}.png"))
                imgs[:,self.past_length+1 +i] = psuedo_img
                depths[:,self.past_length+1 +i] = psuedo_depth[:,0]
            else:
                break
        return loss,loss_dict,imgs,depths


    def pose_loss(self,pred,gt,sim_weight= 1,l1_weight= 1):
        sim_loss = torch.dist(pred,gt) ** 2
        l1_loss = (pred-gt).abs().mean()
        return sim_weight*sim_loss,l1_weight*l1_loss

    def forward_pose(self,batch,loss = 0, loss_dict = dict(),known_mat = None):
        
        known_mat = batch["relative_poses"][:self.past_length+1] if known_mat is None or self.training else known_mat
        _,frontiers_down,sample_points,_ = self.get_frontier(batch)
        outputs = self.posenet(batch,frontiers_down,sample_points,known_mat)
        
        pred_axis_angle,pred_trans = outputs["axis_angle"],outputs["trans"]
        gt_axis_angle,gt_trans = batch["axis_angle"],batch["trans"]
        sim_loss_angle,l1_loss_angle = self.pose_loss(pred_axis_angle,gt_axis_angle,sim_weight= 100,l1_weight= 100)
        sim_loss_trans,l1_loss_trans = self.pose_loss(pred_trans,gt_trans)#,sim_weight = 0.0

        loss = loss + sim_loss_angle + l1_loss_angle + sim_loss_trans + l1_loss_trans
        loss_dict["sim_loss_angle"] = sim_loss_angle.detach()
        loss_dict["l1_loss_angle"] = l1_loss_angle.detach()
        loss_dict["sim_loss_trans"] = sim_loss_trans.detach()
        loss_dict["l1_loss_trans"] = l1_loss_trans.detach()
        #print(pred_axis_angle[0],gt_axis_angle[0])
        #print(pred_trans[0],gt_trans[0])
        pred_mat = transformation_from_parameters(pred_axis_angle.detach().clone(),pred_trans.detach().clone(), invert=False)
        pred_mat = [pred_mat[:,i] for i in range(pred_mat.shape[1])]
        
        
        combined_mat = known_mat + pred_mat
        combined_mat = [p.float() for p in combined_mat]
        return loss,loss_dict,combined_mat

    def forward(self,  batch,step_type = "train", *args, **kwargs):
        
        loss,loss_dict = 0,dict()
        not_train =  (step_type == "val" or step_type == "test")
        if self.mode == "tempnet":
            loss,loss_dict,cond = self.forward_tempnet(batch,loss= loss,loss_dict = loss_dict,step_type =step_type)
        else:
            cond = {}
            if self.train_psuedo or not_train:
                loss,loss_dict,pred_poses = self.forward_pose(batch,loss= loss,loss_dict = loss_dict)
                batch["pred_poses"] = pred_poses
            if self.train_pose or not_train:
                loss,loss_dict,psuedo_imgs,psuedo_depth = self.forward_synth(batch,loss= loss,loss_dict = loss_dict,step_type =step_type)
                batch["psuedo_imgs"] = psuedo_imgs
            if not_train:
                with torch.no_grad():
                    if "voxel_label" in batch:
                        _,frontiers_down,sample_points,sample_depths = self.get_frontier(batch, pred_poses = pred_poses,psuedo_depth = psuedo_depth)
                        outputs = self.tempnet(batch,frontiers_down,sample_points,sample_depths,psuedo_imgs = psuedo_imgs) #
                        y_true  = batch["voxel_label"].long()
                        y_pred = outputs["ssc_pred"].softmax(1).argmax(1)
                        self.update_metric(y_pred,y_true)
        return cond,loss,loss_dict

    def all_gather(self,x_):
        x = all_gather(x_)#.flatten()
        x = [p.to(x_.device) for p in x]
        x = torch.cat(x).cpu().numpy()
        return x

    def update_metric(self,y_pred,y_true):
        preds = self.all_gather(y_pred)
        gts = self.all_gather(y_true)
        self.metrics["val"].add_batch( preds, gts)
        self.metrics["val_n"].add_batch( preds[:,:64,96:160], gts[:,:64,96:160])
        self.metrics["val_m"].add_batch(preds[:,:128,64:192], gts[:,:128,64:192])
        self.metrics["val_f"].add_batch( preds[:,170:], gts[:,170:])
    @classmethod
    def init_from_path(cls_ ,path):
        with open(os.path.join(path,"conf.yaml"), "r") as f:
            config = edict(yaml.safe_load(f))
        generator  = cls_(config)
        weight  = torch.load(os.path.join(path,"checkpoints","last.ckpt"), map_location='cpu')["model"]
        mis,uxp = generator.load_state_dict(weight,strict = False)
        print("load form:",path ,mis,uxp)
        return generator
        
    def configure_optimizers(self):
        param_dicts = [ 
            {"params": [p for n, p in self.named_parameters() if "net_rgb" not in n and p.requires_grad]},
            {
            "params": [p for n, p in self.named_parameters() if "net_rgb" in n and p.requires_grad],
            "lr": self.lr/10,
            },
            ]#backbone.layer1
        #print(param_dicts[1])
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=0.1)
        return optimizer, scheduler

def compute_cascade_matrix_products(matrix_list):
    
    result_list = []
    for i in range(len(matrix_list)):
        current_matrix = matrix_list[i]
        product_result = current_matrix.clone()
        for j in range(i+1, len(matrix_list)):
            product_result = torch.bmm(product_result, matrix_list[j])
        result_list.append(product_result)
    
    return result_list

def mat_to_axisAngle_and_trans(mat):
    rot,trans = mat[...,:3,:3],mat[...,:3,3]
    axis_angle = rotMat_to_axisAngle(rot)
    return axis_angle,trans
def rotMat_to_axisAngle(rot):
    """
    将旋转矩阵转换为轴角表示 (支持batch维度)
    
    参数:
        rot: 形状为 (..., 3, 3) 的旋转矩阵张量
    
    返回:
        形状为 (..., 3) 的轴角向量张量
    """
    # 计算迹 (trace)
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]
    # 限制trace范围以避免数值不稳定
    trace = torch.clamp(trace, 0.0, 2.99999)
    
    # 计算旋转角度
    theta = torch.acos((trace - 1.0) / 2.0)
    
    # 计算反对称矩阵部分
    omega_cross = (theta.unsqueeze(-1).unsqueeze(-1) / 
                  (2 * torch.sin(theta).unsqueeze(-1).unsqueeze(-1))) * (rot - rot.transpose(-1, -2))
    
    # 提取轴角向量
    axis_angle = torch.stack([
        omega_cross[..., 2, 1], 
        omega_cross[..., 0, 2], 
        omega_cross[..., 1, 0]
    ], dim=-1)
    
    return axis_angle
class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def generate_guassian_depth_target(depth, stride, cam_depth_range, constant_std=None):
    depth = depth.flatten(0, 1)
    B, tH, tW = depth.shape
    kernel_size = stride
    center_idx = kernel_size * kernel_size // 2
    H = tH // stride
    W = tW // stride
    
    unfold_depth = F.unfold(depth.unsqueeze(1), kernel_size, dilation=1, padding=0, stride=stride) #B, Cxkxk, HxW
    unfold_depth = unfold_depth.view(B, -1, H, W).permute(0, 2, 3, 1).contiguous() # B, H, W, kxk
    valid_mask = (unfold_depth != 0) # BN, H, W, kxk
    
    if constant_std is None:
        valid_mask_f = valid_mask.float() # BN, H, W, kxk
        valid_num = torch.sum(valid_mask_f, dim=-1) # BN, H, W
        valid_num[valid_num == 0] = 1e10
        
        mean = torch.sum(unfold_depth, dim=-1) / valid_num
        var_sum = torch.sum(((unfold_depth - mean.unsqueeze(-1))**2) * valid_mask_f, dim=-1) # BN, H, W
        std_var = torch.sqrt(var_sum / valid_num)
        std_var[valid_num == 1] = 1 # set std_var to 1 when only one point in patch
    else:
        std_var = torch.ones((B, H, W)).type_as(depth).float() * constant_std

    unfold_depth[~valid_mask] = 1e10
    min_depth = torch.min(unfold_depth, dim=-1)[0] #BN, H, W
    min_depth[min_depth == 1e10] = 0
    
    # x in raw depth 
    x = torch.arange(cam_depth_range[0] - cam_depth_range[2] / 2, cam_depth_range[1], cam_depth_range[2])
    # normalized by intervals
    dist = Normal(min_depth / cam_depth_range[2], std_var / cam_depth_range[2]) # BN, H, W, D
    cdfs = []
    for i in x:
        cdf = dist.cdf(i)
        cdfs.append(cdf)
    
    cdfs = torch.stack(cdfs, dim=-1)
    depth_dist = cdfs[..., 1:] - cdfs[...,:-1]
    
    return depth_dist, min_depth

def transformation_from_parameters(axisangle, translation, invert=False):
    """Convert the network's (axisangle, translation) output into a 4x4 matrix
    """
    R = rot_from_axisangle(axisangle)
    t = translation.clone()
    if invert:
        R = R.transpose(2, 3)
        t *= -1

    T = get_translation_matrix(t)

    if invert:
        M = torch.matmul(R, T)
    else:
        M = torch.matmul(T, R)
    return M


def get_translation_matrix(translation_vector):
    """Convert a translation vector into a 4x4 transformation matrix
    """
    T = torch.zeros(translation_vector.shape[0],translation_vector.shape[1], 4, 4).to(device=translation_vector.device)
    t = translation_vector.contiguous()

    T[:,:, 0, 0] = 1
    T[:,:, 1, 1] = 1
    T[:,:, 2, 2] = 1
    T[:,:, 3, 3] = 1
    T[:,:, :3, 3] = t

    return T


def rot_from_axisangle(vec):
    """Convert an axisangle rotation into a 4x4 transformation matrix
    (adapted from https://github.com/Wallacoloo/printipi)
    Input 'vec' has to be Bx1x3
    """
    angle = torch.norm(vec, 2, 2, True)
    axis = vec / (angle + 1e-7)
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    C = 1 - ca

    x = axis[..., 0].unsqueeze(-1) #b,n,1
    y = axis[..., 1].unsqueeze(-1)
    z = axis[..., 2].unsqueeze(-1)

    xs = x * sa
    ys = y * sa
    zs = z * sa
    xC = x * C
    yC = y * C
    zC = z * C
    xyC = x * yC
    yzC = y * zC
    zxC = z * xC

    rot = torch.zeros((vec.shape[0],vec.shape[1], 4, 4)).to(device=vec.device)
    rot[:,:, 0:1, 0] = x * xC + ca#torch.squeeze(x * xC + ca)
    rot[:,:, 0:1, 1] = xyC - zs#torch.squeeze(xyC - zs)
    rot[:,:, 0:1, 2] = zxC + ys#torch.squeeze(zxC + ys)
    rot[:,:, 1:2, 0] = xyC + zs#torch.squeeze(xyC + zs)
    rot[:,:, 1:2, 1] = y * yC + ca#torch.squeeze(y * yC + ca)
    rot[:,:, 1:2, 2] = yzC - xs#torch.squeeze(yzC - xs)
    rot[:,:, 2:3, 0] = zxC - ys#torch.squeeze(zxC - ys)
    rot[:,:, 2:3, 1] = yzC + xs#torch.squeeze(yzC + xs)
    rot[:,:, 2:3, 2] = z * zC + ca#torch.squeeze(z * zC + ca)
    rot[:,:, 3, 3] = 1

    return rot