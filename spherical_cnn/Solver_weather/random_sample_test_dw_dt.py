import xarray as xr
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt
from pic_util import *
file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
g = 9.80665


time = ds.time.values
level = 850
diffusion_coefficient_flat = 10e-5
diffusion_coefficient_vertical = 1
residuals = []

for time_index in range(0,100,2):
    time_curr = ds.time.values[time_index]
    time_next = ds.time.values[time_index + 1]
    ds_curr = ds.sel(time=time_curr)
    ds_next = ds.sel(time=time_next)
    util_curr = get_point_parameters(ds_curr)
    util_next = get_point_parameters(ds_next)



    ##########u_advection##########
    dwu_dx = util_curr.d_x(level=850, wind_type ="wu")
    dwv_dy = util_curr.d_y(level=850, wind_type ="wv")
    dww_dz = util_curr.d_z(level=[850,925], wind_type ="ww")
    w_advection = -(dwu_dx + dwv_dy + dww_dz)
    ##########u_advection##########

    ##########PGF##########
    dp_dz = util_curr.d_z(level=[850,875], wind_type ="p")
    # dp_dz = util_curr.compute_dp_dz(level= [850, 925])
    rho = util_curr.get_rho(level=850)
    PGF = -( 1 / rho) * dp_dz
    ##########PGF##########

    ##########coriolis_force##########
    coriolis = util_curr.get_w_coriolis_force(level=850)
    ##########coriolis_force##########

    ##########diffusion##########
    dw_dx = util_curr.d_x(level=850, wind_type="w")
    dw_dy = util_curr.d_y(level=850, wind_type="w")
    dw_dz = util_curr.d_z(level=[825, 875], wind_type="w")
    dw_dz_2 = util_curr.d_z(level=[825, 900], wind_type="w")
    dw_ddx = util_curr.dd_x(dx=dw_dx, level=850)
    dw_ddy = util_curr.dd_y(dy=dw_dy, level=850)
    dw_ddz = util_curr.dd_z(level=[825, 875], du1=dw_dz, du2=dw_dz_2)
    diffusion = diffusion_coefficient_flat * (dw_ddx + dw_ddy) + diffusion_coefficient_vertical * dw_ddz
    ##########diffusion##########

    ##########dudt##########
    gravity_term = -g * np.ones_like(w_advection)
    lon = util_curr.get_lon(level=850)
    lat = util_curr.get_lat(level=850)
    dw_dt_exp = w_advection  + diffusion + gravity_term + coriolis + PGF

    ##########dudt##########

    ##########dudt true##########
    w_curr = util_curr.get_wind_w(level=850)
    w_next = util_next.get_wind_w(level=850)
    dw_dt_true = (w_next - w_curr) / 3600
    ##########dudt true##########


    if time_index == 98:
        draw(dw_dt_exp, lon=lon, lat=lat,scale=1,title="dw_dt")
        draw(dw_dt_true, lon=lon, lat=lat,scale=1,title="dw_dt")
    residual = dw_dt_true - dw_dt_exp
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