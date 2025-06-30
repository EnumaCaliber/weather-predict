import xarray as xr
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt
from pic_util import *
import math
import torch
from spherical_cnn.Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels,weighted_rmse_torch
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
v_hour_list = []
v_next_list = []


time = ds.time.values
level = 850
diffusion_coefficient_flat = 10e-5
diffusion_coefficient_vertical = 1
residuals = []
acc_list = []
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
    # PGF = -( 1 / rho) * dp_dy
    PGF = util_curr.get_geopotential_dy(level=850)
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
    ps = util_curr.get_surface_pressure(level=850)
    ##########dudt##########

    # z_surface = ds_curr.sel(level=1000)["geopotential"].values / 9.80665
    # z_level = ds_curr.sel(level=level)["geopotential"].values / 9.80665  # shape: (lon, lat)
    # terrain_mask = (z_level - z_surface) > 200
    # dv_dt_exp = np.where(terrain_mask, dv_dt_exp, np.nan)

    ##########dudt true##########
    v_curr = util_curr.get_wind_v(level=850)
    v_next = util_next.get_wind_v(level=850)
    dv_dt_true = (v_next - v_curr) / 3600
    ##########dudt true##########

    residual = dv_dt_true - dv_dt_exp
    residuals.append(residual.flatten())

    v_pre = v_curr + dv_dt_exp * 3600
    v_hour_list.append(v_pre)  # shape: [1, 1, H, W]
    v_next_list.append(v_next)  # shape: [1, 1, H, W]

    if time_index == 98:
        lon = util_curr.get_lon(level=850)
        lat = util_curr.get_lat(level=850)
        v_hour = v_curr + dv_dt_exp * 3600
        draw(v_next, lon=lon, lat=lat,scale=1,title="v_next")
        draw(v_hour, lon=lon, lat=lat,scale=1,title="v_hour")

residuals_all = np.concatenate(residuals)
v_pre_np = np.stack(v_hour_list, axis=0)   # shape: [N, 240, 121]
v_next_np = np.stack(v_next_list, axis=0)


u_pre_tensor = torch.from_numpy(v_pre_np[:, None, :, :]).float()
u_next_tensor = torch.from_numpy(v_next_np[:, None, :, :]).float()
acc_result = weighted_acc_torch_channels(u_pre_tensor, u_next_tensor)
rmse = weighted_rmse_torch(u_pre_tensor, u_next_tensor)
print("acc_result:", acc_result)
print("rmse:", rmse)


plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
plt.title("Histogram of Residuals: $dv/dt_{true} - dv/dt_{exp}$", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()