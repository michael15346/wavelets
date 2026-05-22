import numpy as np

from offset_tensor import OffsetTensor


def get_adjugate(M):
    Madj = np.rint(np.linalg.det(M) * np.linalg.inv(M)).astype(int)
    return Madj


# получение размеров фундаментальной плитки
# возврат numpy-строки натуральных чисел
def get_tile(M):
    Madj = get_adjugate(M)
    m = np.rint(np.abs(np.linalg.det(M))).astype(int)
    # получение знаменателей после сокращения дробей  Madj_ij / m:
    pre_tile = m // np.gcd(Madj,m)
    # НОК от столбцов через np.lcm.reduce
    tile = np.array([np.lcm.reduce(vec) for vec in pre_tile.T])
    return tile


def get_pad_up_to(shape, M):
    tile = get_tile(M)
    padded_shape = ((np.array(shape) + tile - 1) // tile) * tile
    return padded_shape


# координаты M-кратных точек
# предполагается, что shape является M-кратным
def get_M_multiples_slices(shape, offset, M):
    Madj = get_adjugate(M)
    m = np.rint(np.abs(np.linalg.det(M))).astype(int)
    tile = get_tile(M)
    d = M.shape[0]
    # получение всех точек внутри фундаментальной плитки tile
    in_tile_coords = np.indices(tile).reshape(d, -1)

    mask = np.all(np.mod(Madj @ in_tile_coords, m) == 0, axis=0)
    M_vectors = np.mod((in_tile_coords[:, mask] - offset.reshape(-1, 1)).T, tile)

    tile_shifts = np.array(shape) // tile
    M_multiples_slices = [tuple(slice(M_vec[idx], M_vec[idx] + tile[idx] * ts, tile[idx])
                                for idx, ts in enumerate(tile_shifts))
                          for M_vec in M_vectors]
    return M_multiples_slices

def downsample_fastest(a: OffsetTensor, M: np.ndarray):
    M_multiples_slices = get_M_multiples_slices(a.tensor.shape, a.offset, M)
    return np.array([a.tensor[slc] for slc in M_multiples_slices])

def upsample_fastest(a, M: np.ndarray, original_shape, original_offset, M_multiples_slices = None):
    upsampled = OffsetTensor(np.zeros(original_shape, dtype=np.float64), np.array(original_offset))
    if M_multiples_slices is None:
        M_multiples_slices = get_M_multiples_slices(original_shape, original_offset, M)
    for idx in range(len(M_multiples_slices)):
        upsampled.tensor[M_multiples_slices[idx]] = a[idx, :, :]
    return upsampled
