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



    ##########u_advection##########
    duu_dx = util_curr.d_x(level=850, wind_type ="uu")
    duv_dy = util_curr.d_y(level=850, wind_type ="uv")
    duw_dz = util_curr.d_z(level=[850,975], wind_type ="uw")
    u_advection = -(duu_dx + duv_dy + duw_dz)
    ##########u_advection##########

    ##########PGF##########
    dp_dx = util_curr.d_x(level=850, wind_type ="p")
    rho = util_curr.get_rho(level=850)
    PGF = -( 1 / rho) * dp_dx
    ##########PGF##########

    ##########coriolis_force##########
    coriolis = util_curr.get_u_coriolis_force(level=850)
    ##########coriolis_force##########

    ##########diffusion##########
    du_dx = util_curr.d_x(level=850, wind_type="u")
    du_dy = util_curr.d_y(level=850, wind_type="u")
    du_dz = util_curr.d_z(level=[850, 875], wind_type="u")
    du_dz_2 = util_curr.d_z(level=[850, 900], wind_type="u")
    du_ddx = util_curr.dd_x(dx=du_dx, level=850)
    du_ddy = util_curr.dd_y(dy=du_dy, level=850)
    du_ddz = util_curr.dd_z(level=[850, 875], du1=du_dz, du2=du_dz_2)
    diffusion = diffusion_coefficient_flat * (du_ddx + du_ddy) + diffusion_coefficient_vertical * du_ddz
    ##########diffusion##########

    ##########dudt##########
    du_dt_exp = u_advection + PGF + coriolis + diffusion
    ##########dudt##########

    ##########dudt true##########
    u_curr = util_curr.get_wind_u(level=850)
    u_next = util_next.get_wind_u(level=850)
    du_dt_true = (u_next - u_curr) / 3600
    ##########dudt true##########

    residual = du_dt_true - du_dt_exp
    residuals.append(residual.flatten())


    ##########high different ######
    high_1 = util_curr.get_high_meter(level=850)
    high_2 = util_curr.get_high_meter(level=925)
    print(high_1 - high_2)


residuals_all = np.concatenate(residuals)
plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
plt.title("Histogram of Residuals: $du/dt_{true} - du/dt_{exp}$", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()