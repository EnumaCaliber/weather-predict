import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

from spherical_cnn.Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels,weighted_rmse_torch
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import torch

file_path = "era5_day_2021-01-01.nc"
ds = xr.open_dataset(file_path)


u = ds['u_component_of_wind'].sel(level=500).values  # shape: (24, lat, lon)
T_old, lat, lon = u.shape


dt = 3600
du_dt = (u[1:, :, :] - u[:-1, :, :]) / dt


t_old = np.linspace(0, 22, 23)
t_new = np.linspace(0, 22, (23 - 1) * 6 + 1)


du_dt_interp = np.zeros((len(t_new), lat, lon), dtype=np.float32)

for i in range(lat):
    for j in range(lon):
        series = du_dt[:, i, j]
        if np.any(np.isnan(series)):
            series = np.nan_to_num(series, nan=0.0)
        f = interp1d(t_old, series, kind='cubic')
        du_dt_interp[:, i, j] = f(t_new)

u0 = du_dt_interp[0]


u_reconstructed = np.zeros((du_dt_interp.shape[0] + 1, *du_dt_interp.shape[1:]), dtype=np.float32)
u_reconstructed[0] = u0

for t in range(1, u_reconstructed.shape[0]):
    u_reconstructed[t] = u_reconstructed[t - 1] + 600 * du_dt_interp[t - 1]

from pic_util import *


lon = ds['longitude'].values
lat = ds['latitude'].values



draw(u_reconstructed[-1], lon=lon, lat=lat,scale=1,title="u_integer")
draw(u[-1], lon=lon, lat=lat,scale=1,title="u_true")




u_pre_tensor = torch.from_numpy(u_reconstructed[-1]).unsqueeze(0).unsqueeze(0)
u_next_tensor = torch.from_numpy(u[-1]).unsqueeze(0).unsqueeze(0)



acc_result = weighted_acc_torch_channels(u_pre_tensor, u_next_tensor)
rmse = weighted_rmse_torch(u_pre_tensor, u_next_tensor)

print(acc_result)
print(rmse)