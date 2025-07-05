import numpy as np
import torch
import matplotlib.pyplot as plt
from spherical_cnn.Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch
from scipy.interpolate import CubicHermiteSpline
from pic_util import *
import xarray as xr
data = np.load("du_dt_and_u1000.npz")
file_path = "era5_day_2021-01-01.nc"
ds = xr.open_dataset(file_path)

lon = ds["longitude"].values
lat = ds["latitude"].values


du_dt = data['du_dt']
u_true =  data['u']
t_old = np.arange(24)

lon_n,lat_n = u_true.shape[1:]
u_pred = np.zeros_like(du_dt)
u_pred[0] = u_true[0]

for i in range(lon_n):
    for j in range(lat_n):
        u_series = u_true[:, i, j]
        du_series = du_dt[:, i, j]

        if np.any(np.isnan(u_series)) or np.any(np.isnan(du_series)):
            continue

        spline = CubicHermiteSpline(t_old, du_series, np.zeros_like(du_series))

        for k in range(1, len(t_old)):
            integral = spline.integrate(t_old[k - 1], t_old[k]) * 3600
            u_pred[k, i, j] = u_pred[k - 1, i, j] + integral


u_pre_tensor = torch.from_numpy(u_pred).unsqueeze(1).float()
u_true_tensor = torch.from_numpy(u_true).unsqueeze(1).float()

acc = weighted_acc_torch_channels(u_pre_tensor, u_true_tensor)
rmse = weighted_rmse_torch(u_pre_tensor, u_true_tensor)

