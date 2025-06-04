

import numpy as np


from spherical_cnn.Solver_weather.weather_util import get_point_parameters
from pic_util import *


file_path = "era5_20200601_12.nc"

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

def dz(wind1, wind2, high_diff):
    return (wind1 - wind2) / (high_diff)

util = get_point_parameters(file_path)



du_dx = util.wind_d_x(level=850, wind_type ="u")
du_dy = util.wind_d_y(level=850, wind_type ="u")
du_dz = util.wind_d_z(level=[850,925], wind_type ="u")


dv_dx = util.wind_d_x(level=850, wind_type ="v")
dv_dy = util.wind_d_y(level=850, wind_type ="v")
dv_dz = util.wind_d_z(level=[850,925], wind_type ="v")

dw_dx = util.wind_d_x(level=850, wind_type ="v")
dw_dy = util.wind_d_y(level=850, wind_type ="v")
dw_dz = util.wind_d_z(level=[850,925], wind_type ="v")

draw(dw_dx)

