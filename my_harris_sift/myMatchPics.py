import numpy as np


def match_features(features1, features2, threshold=1.05):
    print('开始匹配')
    num_features = max(features1.shape[0], features2.shape[0])
    matched = np.zeros((num_features, 2))
    confidence = np.zeros((num_features, 1))

    for i in range(features1.shape[0]):
        dist = np.sqrt(np.sum((features1[i] - features2) ** 2, axis=1))  # 欧拉距离
        smallest = np.min(dist)
        second_smallest = np.partition(dist, 1)[1]
        ratio = smallest / second_smallest
        confidence[i] = 1 / ratio
        matched[i, 0] = i
        matched[i, 1] = np.argmin(dist)

    # 按置信度排序
    order = np.argsort(confidence, axis=0)[::-1, 0]
    confidence = confidence[order, :]
    matched = matched[order, :]

    # 删除低于阈值的匹配
    index_x, index_y = (confidence > threshold).nonzero()
    confidence = confidence[index_x, index_y]
    matched = matched[index_x, :]

    print('匹配完毕')
    return matched, confidence
