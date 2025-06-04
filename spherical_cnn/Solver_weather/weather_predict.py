
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from spherical_cnn.Solver_weather.weather_util import get_point_parameters

# 地球参数

file_path = "era5_20200601_12.nc"
# 读取 ERA5 数据



def d_lon(wind,lon_dis):
    dlon = np.zeros_like(wind)
    dlon = (np.roll(wind, -1, axis=0) - np.roll(wind, 1, axis=0)) / (2 * lon_dis)
    return dlon


def d_lat(wind,lat_dis):
    dlat = np.zeros_like(wind)
    # 中心差分（内部点）
    dlat[:, 1:-1] = (wind[:, 2:] - wind[:, :-2]) / (2 * lat_dis)
    # 前向差分（南边界）
    dlat[:, 0] = (wind[:, 1] - wind[:, 0]) / lat_dis
    # 后向差分（北边界）
    dlat[:, -1] = (wind[:, -1] - wind[:, -2]) / lat_dis
    return dlat



util = get_point_parameters(file_path)

lat_dis = util.get_lat_distance(level=850)
lon_dis = util.get_lon_distance(level=850)
u = util.get_wind_u(level=850)

dudx = d_lon(u, lon_dis)
dudy = d_lat(u, lat_dis)


scale = 1e6

plt.figure(figsize=(14, 5))

# ∂u/∂x 图
plt.subplot(1, 2, 1)
plt.imshow(dudx.T * scale, origin='lower', cmap='bwr')
plt.colorbar(label=f"∂u/∂x × {int(scale)} [1e-6 s⁻¹]")
plt.title("∂u/∂x")

# ∂u/∂y 图
plt.subplot(1, 2, 2)
plt.imshow(dudy.T * scale, origin='lower', cmap='bwr')
plt.colorbar(label=f"∂u/∂y × {int(scale)} [1e-6 s⁻¹]")
plt.title("∂u/∂y")

plt.tight_layout()
plt.show()