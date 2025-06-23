import xarray as xr
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from spherical_cnn.Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch
from spherical_cnn.Solver_weather.weather_util import get_point_parameters
import torch
from pic_util import *

# === Load data ===
file_path = "era5_day_2021-01-01.nc"
ds = xr.open_dataset(file_path)

u = ds['u_component_of_wind'].sel(level=500).values
T_old, lat, lon = u.shape
dt = 3600


du_dt = np.zeros_like(u)
du_dt[:-1] = (u[1:] - u[:-1]) / dt
du_dt[-1] = du_dt[-2]

def integrate_from_velocity(du_dt_interp: np.ndarray, u0: np.ndarray, dt: float) -> np.ndarray:

    T = du_dt_interp.shape[0]
    u_reconstructed = np.zeros((T, *du_dt_interp.shape[1:]), dtype=du_dt_interp.dtype)
    u_reconstructed[0] = u0

    for t in range(1, T):
        m0 = du_dt_interp[t - 1]
        m1 = du_dt_interp[t]
        u_reconstructed[t] = u_reconstructed[t - 1] + dt * ((7 / 12) * m0 + (5 / 12) * m1)

    return u_reconstructed



t_old = np.linspace(0, 23, 24)  # hourly
t_new = np.linspace(0, 23, (24 - 1) * 6 + 1)  # every 10 min

du_dt_interp = np.zeros((len(t_new), lat, lon), dtype=np.float32)

for i in range(lat):
    for j in range(lon):
        u_series = u[:, i, j]
        dudt_series = du_dt[:, i, j]
        if np.any(np.isnan(u_series)) or np.any(np.isnan(dudt_series)):
            u_series = np.nan_to_num(u_series, nan=0.0)
            dudt_series = np.nan_to_num(dudt_series, nan=0.0)
        spline = CubicHermiteSpline(t_old, dudt_series, np.zeros_like(dudt_series))
        du_dt_interp[:, i, j] = spline(t_new)


u0 = u[0]  # initial condition
u_reconstructed = np.zeros((len(t_new) + 1, lat, lon), dtype=np.float32)
u_reconstructed[0] = u0

integrate_from_velocity = integrate_from_velocity(du_dt_interp,u0, 600)



lon_vals = ds['longitude'].values
lat_vals = ds['latitude'].values

draw(integrate_from_velocity[-1], lon=lon_vals, lat=lat_vals, scale=1, title="hermite u")
draw(u[-1], lon=lon_vals, lat=lat_vals, scale=1, title="u_true")

# === Evaluate accuracy ===
u_pre_tensor = torch.from_numpy(integrate_from_velocity[-1]).unsqueeze(0).unsqueeze(0)
u_true_tensor = torch.from_numpy(u[-1]).unsqueeze(0).unsqueeze(0)

acc_result = weighted_acc_torch_channels(u_pre_tensor, u_true_tensor)
rmse = weighted_rmse_torch(u_pre_tensor, u_true_tensor)




print("ACC:", acc_result)
print("RMSE:", rmse)