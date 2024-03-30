from pathlib import Path
import random
import pyrootutils
root = str(pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "README.md"],
    pythonpath=True,
    dotenv=True))
from src.datasets.Grss_config import *
from src.datasets.base import BaseDataset
import numpy as np
from src.data import Data
import torch
import os.path as osp
from src.utils import available_cpu_count, starmap_with_kwargs, \
    rodrigues_rotation_matrix, to_float_rgb, shift_point_cloud,\
    jitter_point_cloud, rotate_perturbation_point_cloud_with_normal, \
    shuffle_points, shuffle_data

from hydra import compose, initialize
import hydra
from hydra.utils import instantiate
from sklearn.cluster import KMeans

__all__ = ['GRSS']

def get_pseudo_labels(point_cloud, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(point_cloud)

    # 获取每个点的聚类标签
    pseudo_labels = kmeans.labels_

    return pseudo_labels

def read_grss_points(path, xyz=True, rgb=True, hs=True, intensity=False, semantic=True, instance=False) -> Data:
    root, path = osp.split(path)
    data_name = 'block_'+ path.split('.')[0].split('_')[1] + '.txt'
    data_path = osp.join(root, data_name)
    # data_path = path
    cloud = np.loadtxt(data_path)
    cloud, _ = shuffle_data(cloud)
    unlabeled_point_indexs = (cloud[:,-1]==0)
    cloud[unlabeled_point_indexs, -1] = -1
    unlabeled_point = cloud[unlabeled_point_indexs]
    pseudo_labels = get_pseudo_labels(unlabeled_point, 5)
    cloud[cloud[:,-1]==20, -1] = 0
    cloud[unlabeled_point_indexs, -1] = pseudo_labels+20
    # cloud = cloud[labeled_points_indexs]

    if path.split('.')[0].split('_')[0] == 'perturbate':
        xyz_data = perturbate_data(cloud)
    elif path.split('.')[0].split('_')[0] == 'shift':
        xyz_data = shift_data(cloud)
    else:
        xyz_data = normal_data(cloud)

    rgb_data = np.array(cloud[:, 3:6],dtype='uint8')
    y_data = np.array(cloud[:,-1], dtype='int')
    o_data = np.array(cloud[:,-1], dtype='int')
    xyz_data = torch.from_numpy(xyz_data) if xyz else None
    rgb_data = to_float_rgb(torch.from_numpy(rgb_data)) if rgb else None
    y_data = torch.from_numpy(y_data) if semantic else None
    o_data = torch.from_numpy(o_data) if instance else None
    hs_data = np.zeros((xyz_data.shape[0], 13),dtype='float')
    hs_data = torch.from_numpy(hs_data) if hs else None
    data = Data(pos=xyz_data, rgb=rgb_data, hs=hs_data, y=y_data, o=o_data)
    return data

def normal_data(cloud):
    xyz_data = np.array(cloud[:, :3],dtype='float32')
    return np.array(xyz_data,dtype='float32')

def perturbate_data(cloud):
    perturbate_xyz = rotate_perturbation_point_cloud_with_normal(cloud[:,:3])
    return np.array(perturbate_xyz,dtype='float32')

def shift_data(cloud):
    shift_xyz = shift_point_cloud(cloud[:,:3])
    return np.array(shift_xyz,dtype='float32')

def read_hs_grss_points(path, xyz=True, rgb=True, intensity=False, intensity_dim=[6,7,8], semantic=True, instance=False, is_val=True):
    if len(intensity_dim)==0:
        print("intensity's dim number at less is one")
        return
    root, path = osp.split(path)
    data_name = 'ple_'+path.split('.').split('.')[1] + '.txt'
    data_path = osp.join(root, data_name)
    cloud = np.loadtxt(data_path)
    intensity_dim = [x+6 for x in intensity_dim]
    intensity_data = cloud[:, intensity_dim]
    intensity_data = torch.from_numpy(intensity_data) if intensity else None
    xyz_data = np.array(cloud[:, :3],dtype='float32')
    rgb_data = np.array(cloud[:, :3],dtype='uint8')
    y_data = np.array(cloud[:,-1], dtype='int64')
    o_data = np.array(cloud[:,-1], dtype='int64')

    xyz_data = torch.from_numpy(xyz_data).device('cuda:1') if xyz else None
    rgb_data = to_float_rgb(torch.from_numpy(rgb_data)).device('cuda:1') if rgb else None
    
    y_data = torch.from_numpy(y_data).device('cuda:1') if semantic else None

    o_data = torch.from_numpy(o_data).device('cuda:1') if instance else None
    data = Data(pos=xyz_data, rgb=intensity_data, y=y_data, o=o_data)

    return data



class GRSS(BaseDataset):
    def __init__(self, *args, fold = [], **kwargs):
        self.fold = fold
        super().__init__(*args, val_mixed_in_train=True,**kwargs)
        
    @property
    def class_names(self) -> list:
        return CLASS_NAME
    
    @property
    def num_classes(self):
        return len(INV_OBJECT_LABEL)
    
    @property
    def all_base_cloud_ids(self):   
        # return {
        #     'train': [f'{name}_{i}' for i in range(7) if i not in  self.fold for name in ['ple', 'perturbate', 'shift']],
        #     'val': [f'{name}_{i}' for i in range(7) if i not in  self.fold for name in ['ple', 'perturbate', 'shift']],
        #     'test': [f'{name}_{i}' for i in self.fold for name in ['ple', 'perturbate', 'shift']]
        # }
        return {
            'train': [f'block_{i}' for i in range(7) if i not in  self.fold],
            'val': [f'block_{i}' for i in range(7) if i not in  self.fold],
            'test': [f'block_{i}' for i in self.fold ]
        }

    @property
    def num_block(self):
        return NUM_BLOCKS

    def download_dataset(self):
        pass

    def read_single_raw_cloud(self, raw_cloud_path):
        return read_grss_points(
            raw_cloud_path, xyz=True, rgb=True, hs=True, semantic=True, instance=False)
    
class Mini_GRSS(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, val_mixed_in_train=True, test_mixed_in_val=True, **kwargs)
        

    @property
    def class_names(self) -> list:
        return CLASS_NAME
    
    @property
    def num_classes(self):
        return len(INV_OBJECT_LABEL)
    
    @property
    def all_base_cloud_ids(self):   
        val_block_ids, test_block_ids, train_block_ids = ["mini"], ["mini"], ["mini"]
        return {
            'train': train_block_ids,
            'val': val_block_ids,
            'test': test_block_ids
        }

    @property
    def num_block(self):
        return NUM_BLOCKS

    def download_dataset(self):
        pass

    def read_single_raw_cloud(self, raw_cloud_path):
        return read_grss_points(
            raw_cloud_path, xyz=True, rgb=True, semantic=True, instance=False)
    
class HS_GRSS(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.intensity_dim = kwargs.get("intensity_dim")
        super().__init__(*args, val_mixed_in_train=True, test_mixed_in_val=True, **kwargs)
        
    @property
    def class_names(self) -> list:
        return CLASS_NAME
    
    @property
    def num_classes(self):
        return len(INV_OBJECT_LABEL)
    
    @property
    def all_base_cloud_ids(self):   
        val_block_ids, test_block_ids, train_block_ids = ["hs_data"], ["hs_data"], ["hs_data"]
        return {
            'train': train_block_ids,
            'val': val_block_ids,
            'test': test_block_ids
        }

    @property
    def num_block(self):
        return NUM_BLOCKS

    def download_dataset(self):
        pass

    def read_single_raw_cloud(self, raw_cloud_path):
        return read_hs_grss_points(
            raw_cloud_path, xyz=True, rgb=True, intensity_dim=self.intensity_dim, semantic=True, instance=False)

def class_to_train_val_test_list(sample_list, sample_count, sampled_list):
    chioced_list = []
    while len(chioced_list) < sample_count:
        num = random.choice(sample_list)
        if num not in chioced_list and num not in sampled_list:
            chioced_list.append(str(num))
    return chioced_list

def random_blockPointCloud(num_block, sample_rate = 0.1):
    
    sample_count = int(num_block * sample_rate)
    val_block_ids = class_to_train_val_test_list(range(num_block), sample_count, [])
    test_block_ids = class_to_train_val_test_list(range(num_block), sample_count, val_block_ids)
    train_block_ids = class_to_train_val_test_list(range(num_block), num_block - sample_count*2, 
                                                   val_block_ids+test_block_ids)
    return val_block_ids, test_block_ids, train_block_ids

    
@hydra.main(version_base="1.2", config_path=root + "/configs", config_name="train.yaml")
def main(cfg):
    datamodule_cfg = cfg.datamodule
    data_root = Path("/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/grss/raw")
    path = data_root / 'block_0.txt'
    data = read_grss_points(path)
    stage='train'
    transform=None
    # pre_transform=datamodule_cfg.pre_transform
    pre_transform = None
    pre_filter=None
    on_device_transform=datamodule_cfg.on_device_train_transform
    save_y_to_csr=datamodule_cfg.save_y_to_csr
    save_pos_dtype=torch.float
    save_fp_dtype=torch.half
    xy_tiling=None
    pc_tiling=None
    val_mixed_in_train=True
    test_mixed_in_val=True
    custom_hash=None
    in_memory=datamodule_cfg.in_memory
    point_save_keys=None
    point_no_save_keys=datamodule_cfg.point_no_save_keys
    point_load_keys=None
    segment_save_keys=None
    segment_no_save_keys=None
    segment_load_keys=None
    # room_fir = '/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/s3dis/raw/Area_1/WC_1'
    # read_s3dis_room(room_fir,xyz=True, rgb=True, semantic=True, instance=False,
    #     xyz_room=False, align=False, is_val=True, verbose=False)
    # path = r"/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/grss/raw/data.txt"
    # read_grss_points(path, xyz=True, rgb=True, intensity=False, intensity_dim=3, semantic=True, instance=False)
    # grss = GRSS(root=data_root,transform=datamodule_cfg,save_y_to_csr=save_y_to_csr,in_memory=in_memory,
    #             pre_transform=pre_transform,point_no_save_keys=point_no_save_keys, on_device_transform=on_device_transform)
    path = data_root / "grss" / "raw" / "block_0.txt"
    data = read_grss_points(path)
    hs_grss =  HS_GRSS(root=data_root,transform=datamodule_cfg,save_y_to_csr=save_y_to_csr,in_memory=in_memory,
                pre_transform=pre_transform,point_no_save_keys=point_no_save_keys, on_device_transform=on_device_transform, 
                intensity_dim = [6,7,8])
    print(hs_grss.intensity_dim)
    # print("GRSS Class the frist one block is :{}".format(grss.__getitem__(0)[0]))
    print("???")

if __name__ == "__main__":
    main()