import xarray as xr
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt

file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)



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



    ##########v_advection##########
    dvu_dx = util_curr.d_x(level=850, wind_type ="vu")
    dvv_dy = util_curr.d_y(level=850, wind_type ="vv")
    dvw_dz = util_curr.d_z(level=[850,875], wind_type ="vw")
    v_advection = -(dvu_dx + dvv_dy + dvw_dz)
    ##########v_advection##########

    ##########PGF##########
    dp_dy = util_curr.d_y(level=850, wind_type ="p")
    rho = util_curr.get_rho(level=850)
    PGF = -( 1 / rho) * dp_dy
    ##########PGF##########

    ##########coriolis_force##########
    coriolis = util_curr.get_v_coriolis_force(level=850)
    ##########coriolis_force##########

    ##########diffusion##########
    dv_dx = util_curr.d_x(level=850, wind_type="v")
    dv_dy = util_curr.d_y(level=850, wind_type="v")
    dv_dz = util_curr.d_z(level=[850,875], wind_type="v")
    dv_dz_2 = util_curr.d_z(level=[875, 900], wind_type="v")
    dv_ddx = util_curr.dd_x(dx=dv_dx, level=850)
    dv_ddy = util_curr.dd_y(dy=dv_dy, level=850)
    dv_ddz = util_curr.dd_z(level=[850,875], du1=dv_dz, du2=dv_dz_2)
    diffusion = diffusion_coefficient_flat * (dv_ddx + dv_ddy) + diffusion_coefficient_vertical * dv_ddz
    ##########diffusion##########

    ##########dudt##########
    dv_dt_exp = v_advection + PGF + coriolis + diffusion
    ##########dudt##########

    ##########dudt true##########
    v_curr = util_curr.get_wind_v(level=850)
    v_next = util_next.get_wind_v(level=850)
    dv_dt_true = (v_next - v_curr) / 3600
    ##########dudt true##########

    residual = dv_dt_true - dv_dt_exp
    residuals.append(residual.flatten())

residuals_all = np.concatenate(residuals)
plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
plt.title("Histogram of Residuals: $dv/dt_{true} - dv/dt_{exp}$", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()