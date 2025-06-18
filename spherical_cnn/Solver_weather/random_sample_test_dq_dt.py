import xarray as xr

from spherical_cnn.Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import numpy as np
import matplotlib.pyplot as plt
from pic_util import *
import torch
import math

file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
time = ds.time.values
level = 850
residuals = []
acc_list = []

q_hour_list = []
q_next_list = []

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
    q_hour = q_curr + v_delta_q * 3600

    q_hour_list.append(q_hour)  # shape: [1, 1, H, W]
    q_next_list.append(q_next)  # shape: [1, 1, H, W]


    residual = dq_dt_true + v_delta_q
    residuals.append(residual.flatten())



q_pre_np = np.stack(q_hour_list, axis=0)   # shape: [N, 240, 121]
q_next_np = np.stack(q_next_list, axis=0)
q_pre_tensor = torch.from_numpy(q_pre_np[:, None, :, :]).float()
q_next_tensor = torch.from_numpy(q_next_np[:, None, :, :]).float()
acc_result = weighted_acc_torch_channels(q_pre_tensor, q_next_tensor)
rmse = weighted_rmse_torch(q_pre_tensor, q_next_tensor)
print("acc_result:", acc_result)
print("rmse:", rmse)
# residuals_all = np.concatenate(residuals)
# plt.figure(figsize=(8, 5))
# plt.hist(residuals_all, bins=100, color='steelblue', edgecolor='black')
# plt.title("Histogram of Residuals: $drho/dt_{true} - drho/dt_{exp}$", fontsize=14)
# plt.xlabel("Residual Value", fontsize=12)
# plt.ylabel("Frequency", fontsize=12)
# plt.grid(True)
# plt.tight_layout()
# plt.show()