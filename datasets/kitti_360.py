import glob
import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import pathlib
import copy
from .helpers import vox2box,vox2pix


SPLITS = {
    'train':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync'),
    'trainval':
    ('2013_05_28_drive_0004_sync', '2013_05_28_drive_0000_sync', '2013_05_28_drive_0010_sync',
     '2013_05_28_drive_0002_sync', '2013_05_28_drive_0003_sync', '2013_05_28_drive_0005_sync',
     '2013_05_28_drive_0007_sync','2013_05_28_drive_0006_sync'),
    'val': ('2013_05_28_drive_0006_sync', ),
    'test': ('2013_05_28_drive_0009_sync', ),
}

def load_poses_kitti360(poses_file):
    """加载 poses.txt，返回 {frame_id: 4x4 位姿} 的字典"""
    poses = {}
    with open(poses_file, 'r') as f:
        for line in f:
            data = list(map(float, line.strip().split()))
            if len(data) == 13:  # frame_id + 3x4 矩阵
                frame_id = int(data[0])
                pose_3x4 = np.array(data[1:]).reshape(3, 4)
                pose_4x4 = np.vstack([pose_3x4, [0, 0, 0, 1]])  # 补全为4x4
                poses[frame_id] = pose_4x4
    return poses


def load_calib_cam_to_pose(calib_file, cam_name='image_00'):
    """加载标定文件中的 T_cam_vehicle（相机到车辆坐标系的变换）"""
    with open(calib_file, 'r') as f:
        for line in f:
            if line.startswith(cam_name):
                data = list(map(float, line.strip().split()[1:]))  # 跳过相机名称
                T_cam_vehicle_3x4 = np.array(data).reshape(3, 4)
                # 补全为 4x4 齐次矩阵
                T_cam_vehicle = np.vstack([T_cam_vehicle_3x4, [0, 0, 0, 1]])
                return T_cam_vehicle
    raise ValueError(f"标定文件中未找到相机 {cam_name} 的变换矩阵！")

def find_closest_pose(image_frame_id, poses_dict):
    """找到与图像帧号最接近的位姿帧号"""
    pose_frame_ids = np.array(list(poses_dict.keys()))
    closest_id = pose_frame_ids[np.argmin(np.abs(pose_frame_ids - image_frame_id))]
    return poses_dict[closest_id]

def extract_frame_id(image_path):
    """从图像文件名中提取帧号（如 '000123.png' → 123）"""
    return int(os.path.basename(image_path).split('.')[0])

def load_aligned_poses_to_images(file_path,image_dir ):
    """将图像与位姿通过帧号对齐，并转换到相机坐标系"""
    # 加载标定矩阵
    poses_dict = load_poses_kitti360(file_path)
    T_cam_vehicle = load_calib_cam_to_pose("/data/datasets/kitti-360/calibration/calib_cam_to_pose.txt")
    T_vehicle_cam = np.linalg.inv(T_cam_vehicle)

    # 处理所有图像
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    aligned_poses = []
    for img_path in image_paths:
        frame_id = extract_frame_id(img_path)
        pose_vehicle = find_closest_pose(frame_id, poses_dict)
        pose_cam =  pose_vehicle@T_cam_vehicle  # T_world_cam = T_vehicle_cam @ T_world_vehicle #T_vehicle_cam @
        aligned_poses.append(pose_cam)
    return aligned_poses
def inverse_pose(pose):
    """计算 4x4 位姿矩阵的逆矩阵"""
    R = pose[:3, :3]  # 旋转部分
    t = pose[:3, 3]   # 平移部分
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = R.T  # 旋转的逆是转置
    inv_pose[:3, 3] = -R.T @ t  # 平移的逆
    return inv_pose
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

class KITTI360(Dataset):
    def __init__(
        self,
        config,color_jitter = None, imageset='train'
    ):
        super().__init__()
        root =  config.data_path
        self.data_root = osp.join(root,"unzips","sscbench-kitti") #data_root 
        self.depth_root = osp.join(root,"depth","sequences")
        self.poses_root = osp.join(root,"poses")
        self.label_root = osp.join(self.data_root,"preprocess","labels")

        self.past_length = config.past_length #4
        self.past_step = config.past_step
        self.future_length = config.future_length #5
        self.future_step = config.future_step # 5

        self.sequences = SPLITS[imageset]
        self.split = imageset
        self.strides = [1, ] + [config.stride,]

        self.vox_origin = np.array((0, -25.6, -2))
        self.voxel_size = 0.2
        self.scene_size = (51.2, 51.2, 6.4)
        self.img_shape = (1408, 376)
        self.img_W = 1408 #1216 #1220#1216#
        self.img_H = 384 #376#368 #370#320
        self.scans = []
        calib = self.read_calib()
        P = calib['P2']
        T_velo_2_cam = calib['Tr']
        proj_matrix = P @ T_velo_2_cam
        self.projected_assets = self.proj_box(P,T_velo_2_cam)
        print(self.split,self.sequences)
        for sequence in self.sequences:
            poses = load_aligned_poses_to_images(
                    os.path.join(self.poses_root, sequence , "poses.txt"),os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00/data_rect')
                )

            glob_path = osp.join(self.data_root, 'data_2d_raw', sequence, 'voxels', '*.bin')
            max_id = len(list(pathlib.Path(os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00/data_rect')).glob('*.png'))) - 1
            print(len(poses),len(list(pathlib.Path(os.path.join(self.data_root, 'data_2d_raw', sequence, 'image_00/data_rect')).glob('*.png'))))
            for voxel_path in glob.glob(glob_path):
                frame_id = os.path.splitext(os.path.basename(voxel_path))[0]
                scan = {
                    'sequence': sequence,
                    'P': P,
                    'T_velo_2_cam': T_velo_2_cam,
                    'proj_matrix': proj_matrix,
                    'voxel_path': voxel_path,
                    "max_id":max_id
                }
                
                scan["poses"] = extract_poses(poses,frame_id,n = self.past_length,k = self.future_length,step_p = self.past_step,step_f = self.future_step)
                self.scans.append(scan)
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def proj_box(self,P2,T_velo_2_cam):
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

    def __len__(self):
        return len(self.scans)

    def getrgb(self,sequence, frame_id):
        path = osp.join(self.data_root, 'data_2d_raw', sequence, 'image_00/data_rect',
                            frame_id + '.png')
        img = Image.open(path).convert("RGB")
        # Image augmentation
        # PIL to numpy
        img = np.array(img, dtype=np.float32, copy=False) / 255.0

        h,w,_ = img.shape
        img = np.pad(img, 
                      pad_width=((0, self.img_H - h),
                                 (0, self.img_W - w),
                                 (0, 0)),  # 保持 c 维度不变
                      mode='constant', constant_values=0)  

        #img = img[:self.img_H, :self.img_W, :]  # crop image
        return  self.transforms(img)
    def getdepth(self,sequence, frame_id):
        path = osp.join(self.depth_root, sequence, frame_id + '.npy')
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

    def __getitem__(self, idx):
        scan = self.scans[idx]
        sequence = scan['sequence']
        P = scan['P']
        T_velo_2_cam = scan['T_velo_2_cam']
        proj_matrix = scan['proj_matrix']

        filename = osp.basename(scan['voxel_path'])
        frame_id = osp.splitext(filename)[0]
        lidar_pts,projected_pixs,fov_masks = self.projected_assets
        data = {
            'frame_id': frame_id,
            'sequence': sequence,
            "lidar_pts":lidar_pts,
            "proj_uvd":projected_pixs,
            "fov_mask":fov_masks,
            "E":T_velo_2_cam,
            "K":P[:3, :3],
        }
        target_1_path = osp.join(self.label_root, sequence, frame_id + '_1_1.npy')
        target = np.load(target_1_path)
        data ["voxel_label"] = target
        temp_frame_id = self.gettempid(frame_id,scan["max_id"])
        if self.depth_root is not None:
            data['depth'] = np.stack([self.getdepth(sequence,f_id) for f_id in temp_frame_id])
        data['img'] = torch.stack([self.getrgb(sequence,f_id) for f_id in temp_frame_id])
             
        if "poses" in scan:
            relative_poses = self.get_rela_pose(scan["poses"]) # lidar coord
            data["relative_poses"] = relative_poses
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
        ndarray_to_tensor(data)
        return data

    @staticmethod
    def read_calib():
        P = np.array([
            552.554261,
            0.000000,
            682.049453,
            0.000000,
            0.000000,
            552.554261,
            238.769549,
            0.000000,
            0.000000,
            0.000000,
            1.000000,
            0.000000,
        ]).reshape(3, 4)

        cam2velo = np.array([
            0.04307104361,
            -0.08829286498,
            0.995162929,
            0.8043914418,
            -0.999004371,
            0.007784614041,
            0.04392796942,
            0.2993489574,
            -0.01162548558,
            -0.9960641394,
            -0.08786966659,
            -0.1770225824,
        ]).reshape(3, 4)
        C2V = np.concatenate([cam2velo, np.array([0, 0, 0, 1]).reshape(1, 4)], axis=0)
        V2C = np.linalg.inv(C2V)
        V2C = V2C[:3, :]

        out = {}
        out['P2'] = P
        out['Tr'] = np.identity(4)
        out['Tr'][:3, :4] = V2C
        return out
def ndarray_to_tensor(data: dict):
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                v = v.astype('float32')
            data[k] = torch.from_numpy(v)