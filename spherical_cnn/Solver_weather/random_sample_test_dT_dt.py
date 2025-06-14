import xarray as xr
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt
from pic_util import *


file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
time = ds.time.values
level = 850
residuals = []
R = 287.0
P = 85000
top_model = 10
radius = 6.371e6
cp = 1004


for time_index in range(0,100,2):
    time_curr = ds.time.values[time_index]
    time_next = ds.time.values[time_index + 1]
    ds_curr = ds.sel(time=time_curr)
    ds_next = ds.sel(time=time_next)
    util_curr = get_point_parameters(ds_curr)
    util_next = get_point_parameters(ds_next)



    ##########dp_dt##########
    p_curr = util_curr.get_true_pressure(level=850)
    p_next = util_next.get_true_pressure(level=850)
    dp_dt = (p_next - p_curr) / 3600
    ##########prho##########

    ##########dTdt true##########
    T_curr = util_curr.get_temperature(level=850)
    T_next = util_next.get_temperature(level=850)
    dT_dt_true = (T_next - T_curr) / 3600
    ##########dTdt true##########

    ##########rtcpp##########
    rtcpp = (T_curr * R / (cp * p_curr)) * dp_dt
    ##########rtcpp##########
    lon = util_curr.get_lon(level=850)
    lat = util_curr.get_lat(level=850)
    if time_index == 98:
        T_hour = T_curr + rtcpp * 3600
        draw(T_hour, lon=lon, lat=lat, scale=1, title="T_hour")
        draw(dT_dt_true, lon=lon, lat=lat, scale=1, title="dT_dt_true")
        draw(rtcpp, lon=lon, lat=lat, scale=1, title="rtcpp")

    residual = dT_dt_true  - rtcpp
    residuals.append(residual.flatten())

residuals_all = np.concatenate(residuals)
plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
plt.title("Histogram of Residuals: $drho/dt_{true} - drho/dt_{exp}$", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()