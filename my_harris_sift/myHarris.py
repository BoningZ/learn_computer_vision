import numpy as np
import cv2


def get_interest_points(image, feature_width):
    h, w = image.shape[:2]
    image = image.astype(float)
    print('开始选取兴趣点')
    k = 4
    t = 0.35
    image = np.pad(image, (k, k), 'edge')
    # 拉普拉斯算子
    x_dev_kernal = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    y_dev_kernal = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])
    x_dev = cv2.filter2D(image, -1, x_dev_kernal)
    y_dev = cv2.filter2D(image, -1, y_dev_kernal)

    Ixx = x_dev ** 2
    Iyy = y_dev ** 2
    IxIy = x_dev * y_dev

    eigen_values_map = np.zeros_like(image)
    for i_x in range(k, h + k):
        for i_y in range(k, w + k):
            A11 = Ixx[i_x - k:i_x + k + 1, i_y - k:i_y + k + 1]
            A21 = IxIy[i_x - k:i_x + k + 1, i_y - k:i_y + k + 1]
            A22 = Iyy[i_x - k:i_x + k + 1, i_y - k:i_y + k + 1]

            A11 = A11.sum()
            A21 = A21.sum()
            A22 = A22.sum()

            A = np.array([[A11, A21], [A21, A22]], dtype=np.float32)
            ret, eigenvalues, eigen_vector = cv2.eigen(A, False)
            if abs(eigenvalues[0]) > t and abs(eigenvalues[1]) > t:
                eigen_values_map[i_x, i_y] = eigenvalues[0] * eigenvalues[1]

    eigen_values_map = eigen_values_map[k:k + h, k:k + w]
    # 极大值
    r = 5
    for i in range(h - r):
        for j in range(w - r):
            window = eigen_values_map[i:i + r, j:j + r]
            if window.sum() == 0:
                continue
            else:
                local_max = np.max(window)
            # 非极大值置零
            max_x, max_y = (window == local_max).nonzero()
            window[:] = 0
            window[max_x, max_y] = local_max
            eigen_values_map[i:i + r, j:j + r] = window

    x, y = (eigen_values_map > 0).nonzero()
    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)

    print('兴趣点选取完毕')
    return y, x
