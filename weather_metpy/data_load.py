import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units

ds = xr.open_dataset("era5_day_2021-01-01.nc").metpy.parse_cf()
ds_ = xr.open_dataset("era5_day_2021-01-01.nc")
# === 变量提取 ===
T = ds['temperature'].sel(time=ds.time[0]).metpy.quantify()  # K
u = ds['u_component_of_wind'].sel(time=ds.time[0]).metpy.quantify()  # m/s
v = ds['v_component_of_wind'].sel(time=ds.time[0]).metpy.quantify()
w = ds['vertical_velocity'].sel(time=ds.time[0]).metpy.quantify()
z = ds['vertical_velocity'].sel(time=ds.time[0]).metpy.quantify()
# === 网格间距（单位: meters） ===
lat = ds['latitude'].values
lon = ds['longitude'].values

 # dx, dy shape: (lat, lon)


temp_advection = mpcalc.advection(T, u=u, v=v,latitude =lat, longitude=lon, x_dim=-2,y_dim=-1,vertical_dim = -3)

print(temp_advection)