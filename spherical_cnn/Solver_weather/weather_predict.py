
import xarray as xr
import numpy as np
# 地球参数
R_earth = 6.371e6  # 地球半径 (m)
Omega = 7.2921e-5  # 自转角速度 (rad/s)
R_gas = 287.0  # 干空气气体常数 (J/kg/K)

# 读取 ERA5 数据
ds = xr.open_dataset("era5_20200601_12.nc")
ds_100 = ds.sel(level=850)
# 数据
u = ds_100["u_component_of_wind"].values
v = ds_100["v_component_of_wind"].values
T= ds_100["temperature"].values
p = ds["surface_pressure"].values

w = ds["vertical_velocity"].values #用来处理垂直的求导

# 纬度 南北
lat = ds["latitude"].values  # shape: (32,)
# 经度 东西
lon = ds["longitude"].values  # shape: (64,)
lat = np.tile(lat[:, np.newaxis], (1, 32))

print("hello world")
# info = util.interpolate_wind_wrf("era5_20200601_12.nc")

