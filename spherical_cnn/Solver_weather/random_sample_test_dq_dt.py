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

for time_index in range(0,100,2):
    time_curr = ds.time.values[time_index]
    time_next = ds.time.values[time_index + 1]
    ds_curr = ds.sel(time=time_curr)
    ds_next = ds.sel(time=time_next)
    util_curr = get_point_parameters(ds_curr)
    util_next = get_point_parameters(ds_next)



    ##########prho##########
    u = util_curr.get_wind_u(level=850)
    v = util_curr.get_wind_v(level=850)
    w = util_curr.get_wind_w(level=850)
    dq_dx = util_curr.d_x(level=850, wind_type ="q")
    dq_dy = util_curr.d_y(level=850, wind_type ="q")
    dq_dz = util_curr.d_z(level=[850,975], wind_type ="q")
    v_delta_q = -(u * dq_dx + v * dq_dy + w * (dq_dz))
    ##########prho##########

    ##########dpdt true##########
    q_curr = util_curr.get_specific_humidity(level=850)
    q_next = util_next.get_specific_humidity(level=850)
    dq_dt_true = (q_next - q_curr) / 3600
    ##########dpdt true##########
    lon = util_curr.get_lon(level=850)
    lat = util_curr.get_lat(level=850)
    if time_index == 98:
        q_hour = q_curr + v_delta_q * 3600
        draw(q_hour, lon=lon, lat=lat, scale=1, title="q_hour")
        draw(dq_dt_true, lon=lon, lat=lat, scale=1, title="dq_dt_true")
        draw(v_delta_q, lon=lon, lat=lat, scale=1, title="v_delta_q")
        draw(dq_dz, lon=lon, lat=lat, scale=1, title="dq_dz")

    residual = dq_dt_true + v_delta_q
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