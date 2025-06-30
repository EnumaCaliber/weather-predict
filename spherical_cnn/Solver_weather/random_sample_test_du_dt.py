import xarray as xr
import torch
from spherical_cnn.Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt
from pic_util import *

file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
import math

time = ds.time.values
level = 500
diffusion_coefficient_flat = 10e-5
diffusion_coefficient_vertical = 1
residuals = []
acc_list = []
u_hour_list = []
u_next_list = []
for time_index in range(0, 100, 2):
    time_curr = ds.time.values[time_index]
    time_next = ds.time.values[time_index + 1]
    ds_curr = ds.sel(time=time_curr)
    ds_next = ds.sel(time=time_next)
    util_curr = get_point_parameters(ds_curr)
    util_next = get_point_parameters(ds_next)

    ##########u_advection##########
    duu_dx = util_curr.d_x(level=level, wind_type="uu")
    duv_dy = util_curr.d_y(level=level, wind_type="uv")
    duw_dz = util_curr.d_z(level=[level, level + 50], wind_type="uw")
    u_advection = -(duu_dx + duv_dy + duw_dz)
    ##########u_advection##########

    ##########PGF##########
    dp_dx = util_curr.d_x(level=level, wind_type="p")
    rho = util_curr.get_rho(level=level)
    # PGF = -(1 / rho) * dp_dx

    dz_dx = util_curr.get_geopotential_dx(level=level)  # shape: (lon, lat)
    PGF = - util_curr.g * dz_dx
    ##########PGF##########

    ##########coriolis_force##########
    coriolis = util_curr.get_u_coriolis_force(level=level)
    ##########coriolis_force##########

    ##########diffusion##########
    du_dx = util_curr.d_x(level=level, wind_type="u")
    du_dy = util_curr.d_y(level=level, wind_type="u")
    du_dz = util_curr.d_z(level=[level, level + 50], wind_type="u")
    du_dz_2 = util_curr.d_z(level=[level + 50, level + 100], wind_type="u")
    du_ddx = util_curr.dd_x(dx=du_dx, level=level)
    du_ddy = util_curr.dd_y(dy=du_dy, level=level)

    du_ddz = util_curr.dd_z(level=[level, level+50], du1=du_dz, du2=du_dz_2)

    # du_ddz = util_curr.dd_z_center(levels=[level - 50, level, level + 50])
    diffusion = diffusion_coefficient_flat * (du_ddx + du_ddy) + diffusion_coefficient_vertical * du_ddz
    ##########diffusion##########

    ##########dudt##########

    du_dt_exp = u_advection + PGF + coriolis + diffusion
    ps = util_curr.get_surface_pressure(level=level)
    du_dt_exp = (ps >= level * 100).astype(float) * du_dt_exp
    ##########dudt##########

    ##########dudt true##########
    u_curr = util_curr.get_wind_u(level=level)
    u_next = util_next.get_wind_u(level=level)
    ##########dudt true##########

    u_pre = u_curr + du_dt_exp * 3600
    u_hour_list.append(u_pre)  # shape: [1, 1, H, W]
    u_next_list.append(u_next)  # shape: [1, 1, H, W]

    #########euler ode##########
    if time_index == 98:
        lon = util_curr.get_lon(level=850)
        lat = util_curr.get_lat(level=850)
        u_hour = u_curr + du_dt_exp * 3600
        draw(u_hour, lon=lon, lat=lat, scale=1, title="u_hour")
        draw(u_next, lon=lon, lat=lat, scale=1, title="u_next")
    #########euler ode##########

    ##########high different ######
    high_1 = util_curr.get_high_meter(level=level)
    high_2 = util_curr.get_high_meter(level=level + 50)
    # print(high_1 - high_2)

    du_dt_true = (u_next - u_curr) / 3600
    residual = du_dt_true - du_dt_exp
    residuals.append(residual.flatten())

# Step 1: Stack into [N, H, W]
u_pre_np = np.stack(u_hour_list, axis=0)  # shape: [N, 240, 121]
u_next_np = np.stack(u_next_list, axis=0)

u_pre_tensor = torch.from_numpy(u_pre_np[:, None, :, :]).float()
u_next_tensor = torch.from_numpy(u_next_np[:, None, :, :]).float()
acc_result = weighted_acc_torch_channels(u_pre_tensor, u_next_tensor)
rmse = weighted_rmse_torch(u_pre_tensor, u_next_tensor)
print("acc_result:", acc_result)
print("rmse:", rmse)
# # === 汇总 ===
residuals_all = np.concatenate(residuals)
# rmse = np.sqrt(np.mean(residuals_all ** 2))
# mean_bias = np.mean(residuals_all)
# avg_acc = np.mean(acc_list)
#
# print("RMSE:", rmse)
# print("Mean Bias:", mean_bias)
# print("Mean ACC:", avg_acc)
#
#
#
mean_bias = np.mean(residuals_all)
# print("RMSE:", rmse)
# print("Mean Bias:", mean_bias)
plt.figure(figsize=(8, 5))
plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
plt.title("Histogram of Residuals: $du/dt_{true} - du/dt_{exp}$", fontsize=14)
plt.xlabel("Residual Value", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.show()
