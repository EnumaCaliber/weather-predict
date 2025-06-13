import xarray as xr
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt
from pic_util import *
file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
g = 9.80665
R = 287.0


time = ds.time.values
level = 850
diffusion_coefficient_flat = 10e-5
diffusion_coefficient_vertical = 1
residuals = []

for time_index in range(0,100,2):
    time_curr = ds.time.values[time_index]
    ds_curr = ds.sel(time=time_curr)
    util_curr = get_point_parameters(ds_curr)




    ##########RHO_cur########## 1.15
    RHO_cur = util_curr.get_rho(level=850)
    ##########RHO_cur##########

    ##########PGF##########
    P_true  = util_curr.get_true_pressure(level=850)
    ##########PGF##########

    T = util_curr.get_temperature(level=850)

    ##########R effect##########
    p_exper = RHO_cur * R * T
    ##########R effect##########

    residual = P_true - p_exper
    lon = util_curr.get_lon(level=850)
    lat = util_curr.get_lat(level=850)

    if time_index == 98:
        draw(p_exper, lon=lon, lat=lat,scale=1,title="p_exper")
        draw(P_true, lon=lon, lat=lat,scale=1,title="P_true")
        draw(residual, lon=lon, lat=lat,scale=1,title="redisual")
    residual = P_true - p_exper
    residuals.append(residual.flatten())

residuals_all = np.concatenate(residuals)
plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
plt.title("Histogram of Residuals: $dw/dt_{true} - dw/dt_{exp}$", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()


####插值
####高度对比
####T0 T1 同高度