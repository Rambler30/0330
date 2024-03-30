from pathlib import Path
import os
from sklearn.neighbors import KDTree
import numpy as np
import torch
from torch.utils.data import Dataset

class School(Dataset):
    def __init__(self, split='train', root='dataset/GDUT', num_point=4096, test_idx=[5],
                 block_size=1.0, sample_rate=1.0, transform=None, fea_dim=6, shuffle_idx=False):
        super().__init__()

        self.root = Path(root)
        self.num_point = num_point
        self.data_path = self.root / "lable/university.txt"
        self.lable_path = self.root / "lable/label.txt"
        if split=='train':
            self.index_path = self.root / "indx/trainindx.txt"
        else:
            self.index_path = self.root / "indx/testindx.txt"
        data = np.loadtxt(str(self.data_path))
        label = np.loadtxt(str(self.lable_path))
        self.data = data
        self.label = label
        self.xyz, self.color = data[:,:3], data[:,3:6]
        self.points = data[:,:6]
        with open(str(self.index_path), 'r') as file:
            my_list = []
            # 逐行读取文件内容
            for line in file:
                # 移除字符串两端的方括号和换行符，并以逗号分隔字符串，得到整数字符串列表
                int_str_list = line.strip('[]\n').split(', ')
                # 将整数字符串列表转换为整数列表，并追加到总列表中
                if int_str_list == ['']:
                    continue
                else:
                    my_list.append([int(item) for item in int_str_list])
        
        list_len = [x.__len__() for x in my_list]
        block_num_points = []
        self.block_list = []
        for i, x in enumerate(list_len):
            if x < self.num_point / 4:
                continue
            self.block_list.append(my_list[i]) 
            block_num_points.append(list_len[i])
        sample_prob = block_num_points / np.sum(block_num_points)
        num_iter = int(np.sum(block_num_points) / self.num_point)

        # 若块内点的数量是num_points的n倍，该块重复n次取点
        batch_idxs = []
        for block_index in range(len(block_num_points)):
            batch_idxs.extend([block_index] * int(np.ceil(sample_prob[block_index] * num_iter)))
        self.batch_idxs = batch_idxs

        self.block_coord_min, self.block_coord_max = [], []
        self.block_size=[]
        for block in self.block_list:
            points = self.xyz[block]
            coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
            self.block_size.append(np.array(coord_max - coord_min))
            self.block_coord_min.append(coord_min), self.block_coord_max.append(coord_max)


    def __len__(self):
        return len(self.batch_idxs)

    def __getitem__(self, idx):
        block_idx = self.batch_idxs[idx]
        points_idx = self.block_list[block_idx]
        points = self.points[points_idx]
        label = self.label[points_idx]
        all_points_num = points.shape[0]
        block_size = np.array(self.block_size[block_idx])

        while(True):
            # to select center points that at least 1024 points are covered in a block size 1m*1m
            center_indx = np.random.choice(all_points_num)
            center = points[center_indx][:3]
            block_min = center - [block_size[0], block_size[1], 0]
            block_max = center + [block_size[0], block_size[1], 0]
            point_idxs = np.where(
                (points[:, 0] >= block_min[0]) & (points[:, 0] <= block_max[0]) & (points[:, 1] >= block_min[1]) & (
                            points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > self.num_point / 4:
                break

        if point_idxs.size >= self.num_point:
            selected_points_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            # do not use random choice here to avoid some pts not counted
            dup = np.random.choice(point_idxs.size, self.num_point - point_idxs.size)
            idx_dup = np.concatenate([np.arange(point_idxs.size), np.array(dup)], 0)
            selected_points_idxs = point_idxs[idx_dup]
        
        # 首先试着使用固定尺度进行训练，看看conv的具体效果。
        # selected_point_idxs = np.random.choice(points_idx, self.num_point, replace=False)
        # center = np.random.choice(points_idx)

        center_point = points[center_indx]
        selected_points = points[selected_points_idxs, :]  # num_point * 6
        # centered points中心化
        centered_points = np.zeros((self.num_point, 3))
        centered_points[:, :2] = selected_points[:, :2] - center_point[:2]
        centered_points[:, 2] = selected_points[:, 2]
        # normalized colors
        normalized_colors = selected_points[:, 3:6] / 255.0
        # normalized points
        normalized_points = selected_points[:, :3] - self.block_coord_min[block_idx] / self.block_coord_max[block_idx] - self.block_coord_min[block_idx]

        current_points = np.concatenate((centered_points, normalized_colors, normalized_points), axis=-1)
        current_labels = label[selected_points_idxs]

        current_points = torch.FloatTensor(current_points)
        current_labels = torch.LongTensor(current_labels)

        return current_points, current_labels


if __name__ == '__main__':
    import transform
    data_root = '/mnt/mountA/cwy/pointcloud/scene_seg/dataset/GDUT'

    num_point, test_area, block_size, sample_rate = 4096, 5, 1.0, 0.01
    train_transform = transform.Compose([transform.RandomRotate(along_z=True),
                                         transform.RandomScale(scale_low=0.8,
                                                               scale_high=1.2),
                                         transform.RandomJitter(sigma=0.01,
                                                                clip=0.05),
                                         transform.RandomDropColor(p=0.8, color_augment=0.0)])
    point_data = School(split='train', root=data_root, num_point=num_point, test_idx=test_area, block_size=block_size, sample_rate=sample_rate, transform=train_transform)
    print('point data size:', point_data.__len__())
    print('point data 0 shape:', point_data.__getitem__(0)[0].shape)
    print('point label 0 shape:', point_data.__getitem__(0)[1].shape)
    import torch, time, random
    manual_seed = 123
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)
    def worker_init_fn(worker_id):
        random.seed(manual_seed + worker_id)
    train_loader = torch.utils.data.DataLoader(point_data, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)
    for idx in range(4):
        end = time.time()
        for i, (input, target) in enumerate(train_loader):
            print('time: {}/{}--{}'.format(i+1, len(train_loader), time.time() - end))
            end = time.time()
