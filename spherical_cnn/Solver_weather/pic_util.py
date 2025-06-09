import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
def draw(pic, lon, lat, scale = 1e6, title=""):

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)

    if lon is not None and lat is not None:
        # 假设 lon 和 lat 是 1D 数组
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        plt.imshow(pic.T * scale, origin='lower', extent=extent, cmap='bwr', aspect='auto')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    else:
        plt.imshow(pic.T * scale, origin='lower', cmap='bwr', aspect='auto')

    plt.colorbar(label=title)
    plt.title(title)
    plt.tight_layout()
    plt.show()

def fill_nan_2d(array: np.ndarray, method: str = 'linear') -> np.ndarray:
    """
    用 griddata 对二维数组中的 NaN 进行插值填充。
    如果线性插值失败，会回退使用最近邻。
    """
    assert array.ndim == 2, "只支持二维数组插值"

    ny, nx = array.shape
    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny))

    nan_mask = np.isnan(array)
    valid_mask = ~nan_mask

    # 提取有效点坐标和值
    points = np.stack([xx[valid_mask], yy[valid_mask]], axis=-1)
    values = array[valid_mask]

    # 需要插值的位置
    interp_points = np.stack([xx[nan_mask], yy[nan_mask]], axis=-1)

    # 线性插值
    array_filled = array.copy()
    interp_values = griddata(points, values, interp_points, method=method)

    # 如果还有 NaN（线性插值失败），再用最近邻补全
    if np.isnan(interp_values).any():
        interp_values_nn = griddata(points, values, interp_points, method='nearest')
        interp_values = np.where(np.isnan(interp_values), interp_values_nn, interp_values)

    # 写回插值结果
    array_filled[nan_mask] = interp_values

    return array_filled