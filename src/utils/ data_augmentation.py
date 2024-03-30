import numpy as np

class Data_Augmentation:
    @staticmethod
    def shuffle_data(data):
        """ Shuffle data and labels.
            Input:
            data: B,N,... numpy array
            label: B,... numpy array
            Return:
            shuffled data, label and shuffle indices
        """
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        return data[idx, ...], idx

    @staticmethod
    def shuffle_points(batch_data):
        """ Shuffle orders of points in each point cloud -- changes FPS behavior.
            Use the same shuffling idx for the entire batch.
            Input:
                BxNxC array
            Output:
                BxNxC array
        """
        idx = np.arange(batch_data.shape[1]) 
        np.random.shuffle(idx)
        return batch_data[:,idx,:]

    @staticmethod
    def rotate_perturbation_point_cloud_with_normal(data, angle_sigma=0.06, angle_clip=0.18):
        """ Randomly perturb the point clouds by small rotations
            Input:
            BxNx6 array, original batch of point clouds and point normals
            angle_sigma: 权重系数, 控制随机角度的大小
            angle_clip:  确定随机角度的上下限(-0.18~0.18)
            Return:
            BxNx3 array, rotated batch of point clouds
        """
        rotated_data = np.zeros(data.shape, dtype=np.float32)
        # 对xyz三个轴方向随机生成一个旋转角度
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        # 根据公式构建三个轴方向的旋转矩阵
        Rx = np.array([[1,0,0],
                    [0,np.cos(angles[0]),-np.sin(angles[0])],
                    [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                    [0,1,0],
                    [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                    [np.sin(angles[2]),np.cos(angles[2]),0],
                    [0,0,1]])
        # 按照内旋方式:Z-Y-X旋转顺序获得整体的旋转矩阵
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = data[:,0:3]

        # 分别对坐标与法向量进行旋转,整体公式应该为: Pt = (Rz * Ry * Rx) * P
        rotated_data[:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)

        return rotated_data

    @staticmethod
    def jitter_point_cloud(data, sigma=0.01, clip=0.05):
        """ Randomly jitter points. jittering is per point.
            Input:
            BxNx3 array, original batch of point clouds
            angle_sigma: 权重系数, 控制随机噪声幅度
            angle_clip:  确定随机噪声的上下限(-0.05~0.05)
            Return:
            BxNx3 array, jittered batch of point clouds
        """
        N, C = data.shape
        assert(clip > 0)
        jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
        jittered_data += data  
        return jittered_data

    @staticmethod
    def shift_point_cloud(data, shift_range=0.1):
        N, C = data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (3,))  # 为单个点云生成一个随机的移动偏差
        data[:, :3] = shifts  # 只对点的前三列（坐标）进行移动
        return data

