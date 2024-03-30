import numpy as np
import cv2
from pathlib import Path
import tifffile
from scipy.stats import norm

# 自定义概率密度函数
def custom_pdf(x, mean, std):
    exponent = -0.5 * ((x - mean) / std) ** 2
    prob_density = np.exp(exponent) / (std * np.sqrt(2 * np.pi))
    return prob_density

def joint_entropy(gradients):
    H = np.zeros(gradients.shape)
    pdf = np.zeros(gradients.shape)
    for i in range(gradients.shape[0]):
        temp = 0
        for j in range(gradients.shape[1]):
            gradients_mean = np.mean(gradients[i][j])
            # gradients_std = np.std(gradients[i][j])
            pdf[i][j] = norm.pdf(gradients[i][j], gradients_mean)
            # pdf[i][j] = custom_pdf(gradients[i][j], gradients_mean, gradients_std)
        for b in range(gradients.shape[0]):
            H[i,:, b] = pdf[i,:, b]*np.log2(pdf[i,:, b])

    return -H

def SSI(h_mul, h_mean_mul, h_sigma, h_mean_sigma, all_mean_sigma, b):
    k1, k2, k3 = 0,0,0
    ssi = np.zeros((b, 1))
    for i in range(b):
        up = (2*h_mul[i]*h_mean_mul+k1)*(2*all_mean_sigma[i]+k2)
        down = (np.square(h_mul[i])+np.square(h_mean_mul)+k1
                )*(np.square(h_sigma[i])+np.square(h_mean_sigma)+k2)
        ssi[i] = up / down
    return ssi

def structuralsimilarity(h):
    band_mean = np.zeros((h.shape[0], h.shape[1]))
    # for i in range(h.shape[0]):
    #     for j in range(h.shape[1]):
    #         if i==j:
    #             continue
    #         band_mean[i][j] = np.mean(h[i][j])
    band_mean = np.mean(h, axis=2)
    row, col, b = h.shape[0], h.shape[1], h.shape[2]
    h_mul = mul(h, row, col, b)
    h_mean_mul = mean_mul(band_mean, row, col)
    h_sigma = sigma(h, row, col, h_mul, b)
    h_mean_sigma = mean_sigma(band_mean, row, col, h_mean_mul)
    all_sigma = all_mean_sigma(row, col, h, h_mul, band_mean, h_mean_mul, b)

    # l = l(h_mutal, h_mean_mutal)
    # c = c(h_sita, h_mean_sita)
    # s = s(h_sita, h_mean_sita, all_mean_sita)

    ssi = SSI(h_mul, h_mean_mul, h_sigma, h_mean_sigma, all_sigma, b)
    return ssi

def mul(data, row, col, b):
    d = np.zeros((b,1))
    for i in range(b):
        d[i] = np.sum(data[:,:,i])
    return 1/(row*col)*d

def mean_mul(data, row, col):
    d = np.sum(data)
    return 1/(row*col)*d

def sigma(data, row, col, mutal, b):
    d = np.zeros((b,1))
    for i in range(b):
        d[i] = np.sum(np.square(data[:,:,i] - mutal[i]))
    return np.sqrt(1/(row*col-1)*d)

def mean_sigma(data, row, col, mutal):
    d = np.sum(np.square(data - mutal))
    return np.sqrt(1/(row*col-1)*d)

def all_mean_sigma(row, col, data, mutal, band_mean, mean_mutal, b):
    d = np.zeros((b, 1))
    for i in range(b):
        d[i] = np.sum((data[:,:,i] - mutal[i])*(band_mean - mean_mutal))
    return np.sqrt(1/(row*col-1)*d)

def minisize(label):
    label_flatten = label.reshape((-1,1))
    class_set = np.unique(label_flatten)
    min_size = label_flatten.shape[0]
    for i, classes in enumerate(class_set):
        class_size = np.where(label_flatten == classes)[0].shape[0]
        if min_size > class_size:
            min_size = class_size
    return min_size

def get_each_class_random_sets(img, label, mini_class_size):
    img_flatten = img.reshape((-1, img.shape[-1]))
    label_flatten = label.reshape((-1, 1))
    random_sets = []
    for i in range(np.unique(label_flatten).shape[0]):
        label_indices = np.argwhere(label_flatten==i)[:, 0]
        random_selection = np.random.choice(label_indices, size=mini_class_size, replace=False)
        temp_img = img_flatten[random_selection]
        temp_label = label_flatten[random_selection]
        random_sets.append(np.concatenate((temp_img, temp_label), axis=1))
    return random_sets

def compute_diff(data, full):
    class_diff = np.zeros((full, full))
    class_top_3 = np.zeros((full, 3))
    for i in range(full):
        for j in range(full):
            if i != j:
                d = np.abs(data[i] - data[j])
                class_diff[i, j, :] = d
            indices = np.argsort()[-3:]
        class_top_3[i,j] = indices

    uniqueBands = []
    for i in range(21):
        temp = class_diff[i, :, :].flatten()
        uniqueBands.append(np.unique(temp))

    USBs = np.unique(np.concatenate(uniqueBands))
    return USBs

def main():
    data = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
    gradients = np.gradient(data)[-1]
    H = joint_entropy(gradients)
    SSI = structuralsimilarity(data)
    root = Path("/mnt/mountA/cwy/pointcloud/superpoint_transformer-master/data/grss/raw")
    path = root / 'hs_label.tif'
    img_label = tifffile.imread(path)
    img, label = img_label[:, :,:-1], img_label[:, :,-1]
    mini_class_size = minisize(label)
    random_sets = get_each_class_random_sets(img, label, mini_class_size)
    roi_sets = [roi_set[:, :-1] for roi_set in random_sets]
    roi_label = [roi_set[:, -1] for roi_set in random_sets]
    random_mean_set = np.mean(roi_sets, axis=1)
    full = len(roi_sets)
    USBs = compute_diff(random_mean_set, full)
    hs_data = img_label[USBs]




if __name__ == '__main__':
    main()