import os
import numpy as np
from torch.utils import data
import yaml
import pathlib
import torch
import copy
from PIL import Image
from torchvision import transforms
from scipy.ndimage import distance_transform_edt
from .helpers import vox2box,vox2pix
def inverse_pose(pose):
    """计算 4x4 位姿矩阵的逆矩阵"""
    R = pose[:3, :3]  # 旋转部分
    t = pose[:3, 3]   # 平移部分
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = R.T  # 旋转的逆是转置
    inv_pose[:3, 3] = -R.T @ t  # 平移的逆
    return inv_pose
def load_poses(file_path):
    """加载 KITTI Odometry 的位姿文件"""
    poses = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            values = list(map(float, line.strip().split()))
            pose = np.array(values).reshape(3, 4)  # 转换为 3x4 矩阵
            pose = np.vstack((pose, [0, 0, 0, 1]))  # 补充为 4x4 齐次矩阵
            poses.append(pose)
    return poses

def rotMat_to_axisAngle(rot):

    trace = rot[0,0] + rot[1,1] + rot[2,2]
    trace = np.clip(trace, 0.0, 2.99999)
    theta = np.arccos((trace - 1.0)/2.0)
    omega_cross = (theta/(2*np.sin(theta)))*(rot - np.transpose(rot))

    return [omega_cross[2,1], omega_cross[0,2], omega_cross[1,0]]

def axisAngle_to_rotMat(omega):
    theta = np.sqrt(omega[0]*omega[0] + omega[1]*omega[1] + omega[2]*omega[2])
    if theta < 1e-8:
        return np.eye(3,3)
    omega_cross = np.stack([0.0, -omega[2], omega[1], omega[2], 0.0, -omega[0], -omega[1], omega[0], 0.0])
    omega_cross = np.reshape(omega_cross, [3,3])

    A = np.sin(theta)/theta
    B = (1.0 - np.cos(theta))/(theta**2)
    C = (1.0 - A)/(theta**2)

    omega_cross_square = np.matmul(omega_cross, omega_cross)
    R = np.eye(3,3) + A * omega_cross + B * omega_cross_square
    return R
'''
def relative_transform(P1, P2):
    """计算从 P1 到 P2 的相对转换矩阵"""
    # T_{1 -> 2} = P2 @ P1^{-1}
    P1_inv = inverse_pose(P1)
    #print(P1_inv,np.linalg.inv(P1))
    T_1_to_2 = P2 @ P1_inv
    return T_1_to_2
'''
def relative_transform(P1, P2):
    """计算从 P1 到 P2 的相对转换矩阵"""
    # T_{1 -> 2} = P2 @ P1^{-1}
    P2_inv = inverse_pose(P2)
    T_1_to_2 = P2_inv @ P1 #P1 @ P2_inv# P2_inv @ P1
    #print(P2_inv @ P1, P1 @ P2_inv)
    return T_1_to_2

def extract_poses(poses, frame_id_str, n = 0,k = 0,step_p = 1,step_f = 1):
    frame_id = int(frame_id_str)
    start_id = max(int(frame_id%step_p), frame_id - int(n*step_p))
    past_poses = poses[start_id:frame_id:step_p]
    #print(frame_id,start_id,range(start_id,frame_id,step_p))
    if len(past_poses)<n:
        past_poses = [poses[0],]* (n  - len(past_poses)) + past_poses


    end_id = min(len(poses), frame_id + 1 + int(k * step_f))
    future_poses = poses[frame_id:end_id:step_f]
    #print(future_poses)
    if len(future_poses) < k+1:
        future_poses = future_poses + [poses[-1],]*(k+1  - len(future_poses))
    
    extracted_poses = past_poses + future_poses
    return extracted_poses

class SemKITTI(data.Dataset):
    def __init__(self, config,color_jitter = None, imageset='train'):
        with open("datasets/semantic-kitti.yaml", 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)
        
        self.config = config
        remapdict = semkittiyaml['learning_map']
        self.learning_map_inv = semkittiyaml["learning_map_inv"]

        maxkey = max(remapdict.keys())
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(remapdict.keys())] = list(remapdict.values())

        remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
        remap_lut[0] = 0  # only 'empty' stays 'empty'.
        self.learning_map = remap_lut
        self.all_pred_poses = config.all_pred_poses if "all_pred_poses" in config else False
        self.mode = config.mode if "mode" in config else  "tempnet"
        self.past_length = config.past_length #4
        self.past_step = config.past_step
        self.future_length = config.future_length #5
        self.future_step = config.future_step # 5

        self.scene_size = (51.2, 51.2, 6.4)#whd
        self.vox_origin = np.array([0, -25.6, -2])#xyz
        self.voxel_size = 0.2  # 0.2m
        self.img_W = 1280 #1216 #1220#1216#
        self.img_H = 384#368 #370#320
        self.strides = [1, ] + [config.stride,]
        #onfig.strides

        self.imageset = imageset
        self.data_path = config.data_path
        if imageset == 'train':
            split = semkittiyaml['split']['train']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        elif imageset == 'trainval':
            split = semkittiyaml['split']['train'] + semkittiyaml['split']['valid']
        else:
            raise Exception('Split must be train/val/test')
        self.scans=[]
        self.projected_assets = list()
        for ii,i_folder in enumerate(split):
            # velodyne path corresponding to voxel path
            print(imageset,i_folder)
            calib = self.read_calib(
                os.path.join(config.data_path, str(i_folder).zfill(2), "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            self.projected_assets.append(self.proj_box(P2,P3,T_velo_2_cam))
            
            if imageset != 'test':
                '''
                poses_pred = load_poses(
                    os.path.join(config.data_path,"pred_poses", str(i_folder).zfill(2) + ".txt")
                )'''
            
                poses = load_poses(
                        os.path.join(config.data_path,"poses", str(i_folder).zfill(2) + ".txt")
                    )
            else:
                poses = load_poses(
                    os.path.join(config.data_path, str(i_folder).zfill(2) , "poses.txt")
                )


            voxel_path = os.path.join(config.data_path, str(i_folder).zfill(2) ,'voxels')
            if (imageset == 'train' or imageset == 'trainval') and self.mode == "psuedo":
                img2_path = os.path.join(config.data_path, str(i_folder).zfill(2) ,'image_2')
                files = list(pathlib.Path(img2_path).glob('*.png'))
            else:
                files = list(pathlib.Path(voxel_path).glob('*.bin')) #.label
            max_id = len(list(pathlib.Path(os.path.join(config.data_path, str(i_folder).zfill(2) ,'image_2')).glob('*.png'))) - 1
            for filename in files:
                frame_id = os.path.splitext(os.path.basename(filename))[0]
                scan = {
                        "frame_id":frame_id,
                        "T_velo_2_cam": T_velo_2_cam,
                        "cam_E":P2[:3,:3],
                        "asset_idx": ii ,
                        "sequence": str(i_folder).zfill(2),
                        "max_id":max_id
                    }
                
                scan["poses"] = extract_poses(poses,frame_id,n = self.past_length,k = self.future_length,step_p = self.past_step,step_f = self.future_step)
                #scan["poses_pred"] = extract_poses(poses_pred,frame_id,n = self.past_length,k = self.future_length,step_p = self.past_step,step_f = self.future_step)
                self.scans.append(scan)
        print(color_jitter)
        self.color_jitter = (
            transforms.ColorJitter(*color_jitter) if color_jitter else None
        )
        self.normalize_rgb = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    def proj_box(self,P2,P3,T_velo_2_cam):
        lidar_pts = list()
        projected_pixs = list()
        fov_masks = list()
        for ii,s in enumerate(self.strides):
            projected_pix, fov_mask, pix_z,lidar_pt  = vox2pix(
                        T_velo_2_cam,
                        P2[0:3, 0:3],
                        self.vox_origin,
                        self.voxel_size *s,
                        self.img_W,
                        self.img_H,
                        self.scene_size,
                    )
            print(lidar_pt.shape)
            uvd = np.concatenate((projected_pix, pix_z[:,None]), axis=-1).astype(np.float32)
            lidar_pts.append(lidar_pt)
            projected_pixs.append(uvd)
            fov_masks.append(fov_mask)
        return lidar_pts,projected_pixs,fov_masks
    def unpack(self, compressed):
        ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
        uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
        uncompressed[::8] = compressed[:] >> 7 & 1
        uncompressed[1::8] = compressed[:] >> 6 & 1
        uncompressed[2::8] = compressed[:] >> 5 & 1
        uncompressed[3::8] = compressed[:] >> 4 & 1
        uncompressed[4::8] = compressed[:] >> 3 & 1
        uncompressed[5::8] = compressed[:] >> 2 & 1
        uncompressed[6::8] = compressed[:] >> 1 & 1
        uncompressed[7::8] = compressed[:] & 1
        return uncompressed
    @staticmethod
    def read_calib(calib_path):
        """
        Modify from https://github.com/utiasSTARS/pykitti/blob/d3e1bb81676e831886726cc5ed79ce1f049aef2c/pykitti/utils.py#L68
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])
        #print(calib_all)
        # reshape matrices
        calib_out = {}
        # 3x4 projection matrix for left camera
        calib_out["P2"] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4)  # 4x4 matrix
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4)
        return calib_out

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.scans)

    def getvox(self,path):
        if self.imageset == 'test':
            voxel_label = np.zeros([256, 256, 32], dtype=int).reshape((-1, 1)).reshape((256, 256, 32))
            return voxel_label
        else:
            voxel_label = np.fromfile(path, dtype=np.uint16).reshape((-1, 1))  # voxel labels
            invalid = self.unpack(np.fromfile(path.replace('label', 'invalid').replace('voxels', 'voxels'), dtype=np.uint8)).astype(np.float32)
        voxel_label = self.learning_map[voxel_label]

        voxel_label = voxel_label.reshape((256, 256, 32))
        invalid = invalid.reshape((256,256,32))
        voxel_label[invalid == 1]=255
        return voxel_label
    def getrgb(self,path):
        img = Image.open(path).convert("RGB")
        # Image augmentation
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        h,w,_ = img.shape
        img = np.pad(img, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w),
                                 (0, 0)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)  
        #img = img[:self.img_H, :self.img_W, :]  # crop image
        return self.normalize_rgb(img)
    def getdepth(self,path):
        depth = np.load(path)
        #depth = depth[:self.img_H, :self.img_W][None]
        h,w = depth.shape
        depth = np.pad(depth, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)

        return depth
    def gettempid(self,frame_id,max_id):
        frame_id = int(frame_id)
        n,k,step_p,step_f = self.past_length,self.future_length,self.past_step,self.future_step

        start_id = max(int(frame_id%step_p), frame_id - int(n*step_p)) #max(0, frame_id - n)
        past_frames = list(range(start_id,frame_id,step_p))
        if len(past_frames)<n:
            past_frames = [0,]* (n  - len(past_frames)) + past_frames

        end_id = min(max_id+1 , frame_id + 1 + int(k*step_f))
        future_frames = list(range(frame_id,end_id,step_f))
        if len(future_frames) < k+1:
            future_frames = future_frames + [max_id,]*(k+1  - len(future_frames))
        frames = past_frames + future_frames
        
        zfill_list = [str(num).zfill(6) for num in frames]
        return zfill_list 
    def get_rela_pose(self,poses):
        #p2 = poses[self.past_length]
        #rela_poses = [relative_transform(p1, copy.deepcopy(p2)) for p1 in poses]
        p1 = poses[self.past_length]
        rela_poses = [relative_transform(copy.deepcopy(p1), p2) for p2 in poses]
        #print(rela_poses)
        return rela_poses
    
    def __getitem__(self, index):
        scan = self.scans[index]
        frame_id = scan["frame_id"]
        sequence = scan["sequence"]
        temp_frame_id = self.gettempid(frame_id,scan["max_id"])
        rgbs = torch.stack([self.getrgb(os.path.join(self.data_path, sequence,'image_2',f_id + ".png")) for f_id in temp_frame_id])
        depths = np.stack([self.getdepth(os.path.join(self.data_path, sequence,'depth2',f_id + ".npy")) for f_id in temp_frame_id])
        lidar_pts,projected_pixs,fov_masks = self.projected_assets[scan["asset_idx"]]
        data = {
            "frame_id": frame_id,
            "img":rgbs,
            "depth":depths,
            "lidar_pts":lidar_pts,
            "proj_uvd":projected_pixs,
            "fov_mask":fov_masks,
            "E":scan["T_velo_2_cam"],
            "K":scan["cam_E"],
            "sequence":scan["sequence"]
        }

        voxel_path = os.path.join(self.data_path, sequence,'voxels',frame_id + ".label")
        if os.path.exists(voxel_path):
            data["voxel_label"] = self.getvox(voxel_path)
        if "poses" in scan:
            relative_poses = self.get_rela_pose(scan["poses"]) # lidar coord
            data["relative_poses"] = relative_poses
            #relative_poses_pred = self.get_rela_pose(scan["poses_pred"])
            #data["relative_poses_pred"] = relative_poses_pred
            #if self.all_pred_poses:
            #    future_poses = relative_poses
            #else:
            future_poses = relative_poses[self.past_length+1:]
            if len(future_poses)> 0 :
                axisangles,trans = list(), list()
                for pose in future_poses:
                    R = pose[0:3,0:3]
                    axisangles.append(np.asarray(rotMat_to_axisAngle(R))) 
                    trans.append(pose[0:3,3])
                axisangles = np.stack(axisangles)
                trans = np.stack(trans)
                data["axis_angle"] = axisangles
                data["trans"] = trans
  
            #data["poses"] = scan["poses"]
        return data 
 

