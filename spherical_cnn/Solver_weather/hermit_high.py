import xarray as xr
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
file_path = "era5_day_2021-01-01.nc"
# 读取 ERA5 数据
ds = xr.open_dataset(file_path)
time_curr = ds.time.values[0]
ds_time = ds.sel(time=time_curr)

def compute_du_dz(z, u):
    du_dz = np.zeros_like(u)
    n = len(z)

    for i in range(n):
        if i == 0:  # forward
            dz = z[i+1] - z[i]
            du_dz[i] = (u[i+1] - u[i]) / dz
        elif i == n - 1:  # backward
            dz = z[i] - z[i-1]
            du_dz[i] = (u[i] - u[i-1]) / dz
        else:  # central
            dz = z[i+1] - z[i-1]
            du_dz[i] = (u[i+1] - u[i-1]) / dz

    return du_dz

def get_u_z_function(z_vals, u_vals):
    valid = (~np.isnan(z_vals)) & (~np.isnan(u_vals))
    if np.sum(valid) < 2:
        return lambda z_query: np.full_like(z_query, np.nan)

    z = z_vals[valid]
    u = u_vals[valid]
    sort_idx = np.argsort(z)
    z = z[sort_idx]
    u = u[sort_idx]

    if len(z) < 2 or np.any(np.diff(z) == 0):
        return lambda z_query: np.full_like(z_query, np.nan)

    du_dz = compute_du_dz(z, u)

    spline = CubicHermiteSpline(z, u, du_dz)
    return spline  # 可调用函数


def build_interp_funcs(z, u):

    level, lon, lat = z.shape
    u_interp_funcs = np.empty((lon, lat), dtype=object)

    for i in range(lon):
        for j in range(lat):
            z_profile = z[:, i, j]
            u_profile = u[:, i, j]
            u_interp_funcs[i, j] = get_u_z_function(z_profile, u_profile)

    return u_interp_funcs

z_t = ds_time["geopotential"].values / 9.80665   # (37, 240, 121)
u_t = ds_time["u_component_of_wind"].values      # (37, 240, 121)

z_t_850 = ds_time.sel(level=850)["geopotential"].values / 9.80665
z_flat = z_t_850.flatten()

plt.hist(z_flat, bins=100)
plt.title("850 hPa Altitude Histogram")
plt.xlabel("Height (m)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()


u_interp_funcs = build_interp_funcs(z_t, u_t)

z_query = 1000.0
lon, lat = z_t.shape[1], z_t.shape[2]
u_at = np.empty((lon, lat))

for i in range(lon):
    for j in range(lat):
        u_at[i, j] = u_interp_funcs[i, j](z_query)

print(u_at)