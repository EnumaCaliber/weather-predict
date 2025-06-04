

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
lat_dis = util.get_lat_distance(level=850)
lon_dis = util.get_lon_distance(level=850)
u1 = util.get_wind_u(level=850)
u2 = util.get_wind_u(level=925)
v1 = util.get_wind_v(level=850)
v2 = util.get_wind_v(level=925)
w1= util.get_wind_w(level=850)
w2 = util.get_wind_w(level=925)
high_diff = util.get_high_diff(level1=850,level2=925)


du_dx = d_lon(u1, lon_dis)
du_dy = d_lat(u1, lat_dis)
du_dz = dz(u1, u2, high_diff)


dv_dx = d_lat(v1, lon_dis)
dv_dy = d_lon(v1, lat_dis)
dv_dz = dz(v1, v2, high_diff)

dw_dx = d_lon(w1, lon_dis)
dw_dy = d_lat(w1, lat_dis)
dw_dz = dz(w1, w2, high_diff)



