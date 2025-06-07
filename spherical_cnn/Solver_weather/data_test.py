
import xarray as xr
import numpy as np
# 地球参数

file_path = "era5_20200601_12_2.nc"
# 读取 ERA5 数据
ds = xr.open_dataset(file_path)
ds_850 = ds.sel(level=850)

# 数据
u = ds_850["u_component_of_wind"].values
v = ds_850["v_component_of_wind"].values
T= ds_850["temperature"].values
p = ds["surface_pressure"].values

w = ds["vertical_velocity"].values #用来处理垂直的求导

# 纬度 南北
lat = ds["latitude"].values  # shape: (32,)
# 经度 东西
lon = ds["longitude"].values  # shape: (64,)
lat = np.tile(lat[:, np.newaxis], (1, 32))
po = ds_850["geopotential"].values / 9.8

print(ds["time"].values)
