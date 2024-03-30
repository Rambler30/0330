import os
import h5py
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


def make_dataset(split='train', data_root=None, data_list=None):
    if not os.path.isfile(data_list):
        raise (RuntimeError("Point list file do not exist: " + data_list + "\n"))
    point_list = []
    list_read = open(data_list).readlines()
    print("Totally {} samples in {} set.".format(len(list_read), split))
    for line in list_read:
        point_list.append(os.path.join(data_root, line.strip()))
    return point_list

class List_to_txt:
    def __init__(self) -> None:
        pass

    @staticmethod
    def read(path):
        with open(path, "r") as file:
            lines = file.readlines()
        data_list = []
        for line in lines:
            values = line.strip().split()  # 使用适当的分隔符拆分数据
            data = [int(value) for value in values]
            data_list.append(data)
        return data_list
    
    @staticmethod
    def write(path, list):
        with open(path, "w") as file:
            for row in list:
                line = " ".join(map(str, row))
                file.write(line + "\n")

class Grss_Val(Dataset):
    def __init__(self, split='val', data_root=None, list_path=None, transform=None,
                 num_point=4096, random_index=False, norm_as_feat=True, fea_dim=6):
        assert split in ['train', 'val', 'test']
        self.split = split
        data_root = Path(data_root)
        self.index_path = data_root / list_path
        self.data_path = data_root / "data/data.txt"
        data = np.loadtxt(self.data_path)
        self.data = data
        self.xyz, self.color, self.lable = data[:, :3], data[:, 3:6], data[:,-1]
        self.points = data[:,:6]
        
        my_list = List_to_txt.read(self.index_path)
        self.data_list = my_list

        self.transform = transform
        self.num_point = num_point
        self.random_index = random_index
        self.norm_as_feat = norm_as_feat
        self.fea_dim = fea_dim
        self.block_coord_min, self.block_coord_max = [], []
        for block in self.data_list:
            points = self.xyz[block]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.block_coord_min.append(coord_min), self.block_coord_max.append(coord_max)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data_index = self.data_list[index]
        if self.num_point is None:
            self.num_point = data_index.__len__()
        if self.random_index:
            np.random.shuffle(data_index)
        data_index = data_index[0: self.num_point]
        center = np.random.choice(data_index)
        center_point = self.points[center,:3]
        selected_points = self.points[data_index, :6]  # num_point * 6
        # centered points中心化
        centered_points = np.zeros((self.num_point, 3))
        centered_points[:, :2] = selected_points[:, :2] - center_point[:2]
        centered_points[:, 2] = selected_points[:, 2]
        # normalized colors
        normalized_colors = selected_points[:, 3:6] / 255.0
        # normalized points
        normalized_points = selected_points[:, :3] - self.block_coord_min[index] / self.block_coord_max[index] - self.block_coord_min[index]

        current_points = np.concatenate((centered_points, normalized_colors, normalized_points), axis=-1)
        current_labels = self.lable[data_index]

        if self.transform is not None:
            current_points, current_labels = self.transform(current_points, current_labels)

        current_points = torch.FloatTensor(current_points)
        current_labels = torch.LongTensor(current_labels)

        return current_points, current_labels


if __name__ == '__main__':
    data_root = '/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GRSS/2018IEEE_Contest/Phase2/TrainingGT'
    data_list = 'idxs/val_index.txt'
    point_data = Grss_Val('train', data_root, data_list)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
