import struct
import torch
import numpy as np

def save_fvecs(filename, data):
    """
    保存数据为 .fvecs 格式。
    :param filename: 输出文件名
    :param data: 一个二维 numpy 数组，形状为 (n, d)
    """
    assert data.ndim == 2, "数据需要是二维数组"
    n, d = data.shape
    # 构建 .fvecs 文件内容
    with open(filename, 'wb') as f:
        for vec in data:
            # 写入向量的维度 (作为 float32)
            f.write(np.array([d], dtype=np.int32).tobytes())
            # 写入向量内容
            f.write(vec.astype(np.float32).tobytes())

def read_fvecs(file_path):
    vectors = []

    with open(file_path, 'rb') as f:
        while True:
            # 读取当前向量的维度（第一个int32整数）
            dim_data = np.fromfile(f, dtype=np.int32, count=1)
            if dim_data.size == 0:
                # 如果读取不到数据，说明文件结束
                break

            dim = dim_data[0]
            # 根据读取的维度，接着读取该行的浮点数数据
            vector = np.fromfile(f, dtype=np.float32, count=dim)

            # 将读取的向量添加到列表
            vectors.append(vector)

    # 将向量列表转换为一个二维数组
    vectors = np.vstack(vectors)
    return vectors

def Read_ivecs(filename):
    """Read .fvecs file and return a list of vectors."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension of the vector (first 4 bytes, int32)
            data = f.read(4)
            if not data:
                break
            dim = struct.unpack('i', data)[0]
            # Read the floats (dim * 4 bytes)
            vector = struct.unpack('i' * dim, f.read(4 * dim))
            vectors.append(vector)
    return np.array(vectors)

def Read_fvecs(filename):
    """Read .fvecs file and return a list of vectors."""
    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension of the vector (first 4 bytes, int32)
            data = f.read(4)
            if not data:
                break
            dim = struct.unpack('i', data)[0]
            # Read the floats (dim * 4 bytes)
            vector = struct.unpack('f' * dim, f.read(4 * dim))
            vectors.append(vector)
    return np.array(vectors)

def Write_ivecs(filename, vectors):
    """Write a list of vectors to .fvecs file."""
    with open(filename, 'wb') as f:
        for vec in vectors:
            if isinstance(vec,torch.Tensor):
                vec=vec.tolist()
            elif isinstance(vec,list) and isinstance(vec[0],torch.Tensor):
                vec=vec[0].tolist()
            # Write dimension
            f.write(struct.pack('i', len(vec)))
            # Write vector data
            f.write(struct.pack('i' * len(vec), *vec))

def Calculate_recall(gt, result, K):
    total_recall = 0.0
    num_queries = len(gt)

    for i in range(num_queries):
        if isinstance(gt[i], list):
            if isinstance(gt[i][0], torch.Tensor):
                gt_list = gt[i][0][:K].tolist()
            else:
                gt_list = gt[i][:K]
        elif isinstance(gt[i], np.ndarray):
            gt_list = gt[i][:K].tolist()
        elif isinstance(gt[i], torch.Tensor):
            gt_list = gt[i][:K].tolist()

        if isinstance(result[i], list):
            if isinstance(result[i][0], torch.Tensor):
                result_list = result[i][0][:K].tolist()
            else:
                result_list = result[i][:K]
        elif isinstance(result[i], np.ndarray):
            result_list = result[i][:K].tolist()
        elif isinstance(result[i], torch.Tensor):
            result_list = result[i][:K].tolist()

        gt_set, result_set = set(gt_list), set(result_list)
        recall = len(gt_set & result_set) / len(gt_set)
        total_recall += recall

    average_recall = total_recall / num_queries
    return average_recall