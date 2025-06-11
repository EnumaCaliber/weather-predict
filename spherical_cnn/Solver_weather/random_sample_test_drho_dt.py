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
    durho_dx = util_curr.d_x(level=850, wind_type ="uu")

    dvrho_dy = util_curr.d_y(level=850, wind_type ="vrho")
    dwrho_dz = util_curr.d_z(level=[850,975], wind_type ="wrho")
    rv_predict = (durho_dx + dvrho_dy + dwrho_dz)
    ##########prho##########

    ##########dudt true##########
    rho_curr = util_curr.get_rho(level=850)
    print(rho_curr)
    rho_next = util_next.get_rho(level=850)
    drho_dt_true = (rho_next - rho_curr) / 3600
    ##########dudt true##########
    lon = util_curr.get_lon(level=850)
    lat = util_curr.get_lat(level=850)
    if time_index == 98:
        draw(drho_dt_true, lon=lon, lat=lat, scale=1, title="drho_dt_true")
        draw(rv_predict, lon=lon, lat=lat, scale=1, title="rv_predict")

    residual = drho_dt_true - rv_predict
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