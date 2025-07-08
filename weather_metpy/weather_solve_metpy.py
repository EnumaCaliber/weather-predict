import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import os



ds = xr.open_dataset("C:/ProTang/weather-predict/Solver_weather/era5_day_2021-01-01.nc", engine="netcdf4")
ds = ds.metpy.parse_cf(ds)
time = ds.time.values

u = ds['u_component_of_wind'].sel(level = 850,time=time[0]).metpy.quantify().metpy.convert_units('m/s')
v = ds['v_component_of_wind'].sel(level = 850,time=time[0]).metpy.quantify().metpy.convert_units('m/s')
omega = ds['vertical_velocity'].sel(level = 850,time=time[0]).metpy.quantify().metpy.convert_units('pascal/second')
u_next = ds['u_component_of_wind'].sel(level = 850,time=time[2]).metpy.quantify().metpy.convert_units('m/s')


lat = ds['latitude'].values * units.degrees
lon = ds['longitude'].values



adv_horizontal = mpcalc.advection(u, u = u, v=v, x_dim=-2,y_dim=-1,vertical_dim=-3)  # shape: (level, lat, lon)，单位 m/s²
adv_trimmed = adv_horizontal[1:-1, 1:-1]
adv_trimmed_copy = adv_trimmed.copy()
adv_trimmed_copy[0, :] /= 10000
adv_trimmed_copy[-1, :] /= 10000

# 左右列（去掉已经除过的角点）
adv_trimmed_copy[1:-1, 0] /= 10000
adv_trimmed_copy[1:-1, -1] /= 10000


coriolis = mpcalc.coriolis_parameter(lat)


f_xr = xr.DataArray(coriolis, coords={ 'latitude': lat }, dims=["latitude"])
f_2d = f_xr.broadcast_like(v)  # shape 和 v 一致，无需 np.broadcast_to
f_2d = np.broadcast_to(coriolis[np.newaxis:], v.shape)





delta_u = ((adv_trimmed + f_2d * v) * 7200 * units.second)
u_pre = u + delta_u







from pic_util import *
draw(u_pre, lon=lon, lat=lat, scale=1, title="u_hour")
draw(u_next, lon=lon, lat=lat, scale=1, title="u_next")