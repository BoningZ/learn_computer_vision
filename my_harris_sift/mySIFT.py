import numpy as np
import cv2


def get_features(image, x, y, feature_width):
    print('开始计算特征值')
    features = np.zeros((x.shape[0], 128))
    x_ = x.astype(int)
    y_ = y.astype(int)
    y = np.squeeze(x_)
    x = np.squeeze(y_)
    h, w = image.shape[:2]
    # 避免边缘的零
    offset = feature_width // 2
    image = np.pad(image, (offset, offset), 'edge')
    x += offset
    y += offset

    # 高斯模糊
    k = 7
    blur_image = cv2.GaussianBlur(image, (k, k), sigmaX=0)

    # 计算梯度
    x_kernel = np.array([[-1, 1]])
    Gx = cv2.filter2D(blur_image, -1, x_kernel)
    Gy = cv2.filter2D(blur_image, -1, np.transpose(x_kernel))

    # 模与方向
    mag = np.sqrt(Gx ** 2 + Gy ** 2)
    orient = np.arctan2(Gy, Gx)
    orient[orient < 0] += np.pi * 2

    # 高斯函数
    cell_length = feature_width // 4
    Gau_kernel = cv2.getGaussianKernel(feature_width, feature_width / 2)

    for i in range(x.shape[0]):
        window = mag[(x[i] - feature_width // 2):(x[i] + feature_width // 2),
                 (y[i] - feature_width // 2):(y[i] + feature_width // 2)]
        window = cv2.sepFilter2D(window, -1, Gau_kernel, np.transpose(Gau_kernel))
        window_orient = orient[(x[i] - feature_width // 2):(x[i] + feature_width // 2),
                        (y[i] - feature_width // 2):(y[i] + feature_width // 2)]
        for i_x in range(4):
            for i_y in range(4):
                bin = np.zeros(8)
                cell = window[i_x * cell_length:(i_x + 1) * cell_length, i_y * cell_length:(i_y + 1) * cell_length]
                cell_orient = window_orient[i_x * cell_length:(i_x + 1) * cell_length,
                              i_y * cell_length:(i_y + 1) * cell_length]
                for angle in range(bin.shape[0]):
                    bin[angle] += np.sum(
                        cell[np.all([cell_orient >= (angle * np.pi / 4), cell_orient < ((angle + 1) * np.pi / 4)], 0)])
                features[i, (i_x * 4 + i_y) * 8:(i_x * 4 + i_y) * 8 + 8] = bin
        # 归一化
        features[i, :] /= np.sqrt(np.sum(features[i, :] ** 2))

    print('特征值计算完毕')
    return features
