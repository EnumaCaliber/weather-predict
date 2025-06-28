import xarray as xr
import numpy as np
from scipy.interpolate import CubicHermiteSpline
import matplotlib.pyplot as plt
file_path = "era5_day_2021-01-01.nc"

ds = xr.open_dataset(file_path)
time_curr = ds.time.values[0]
ds_time = ds.sel(time=time_curr)


varnames = ["u_component_of_wind", "v_component_of_wind",
            "temperature", "specific_humidity",
            "geopotential","level"]


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

def make_1d_spline(z_vals, u_vals):
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
            u_interp_funcs[i, j] = make_1d_spline(z_profile, u_profile)

    return u_interp_funcs

def build_all_interp_funcs(z3d, ds_time, varnames):
    lev, lon, lat = z3d.shape
    funcs = {v: np.empty((lon, lat), dtype=object) for v in varnames}

    for i in range(lon):
        for j in range(lat):
            z_col = z3d[:, i, j]
            for v in varnames:
                if v == "level":
                    y_col = ds_time.level.values  # shape (37,)
                else:
                    y_col = ds_time[v].values[:, i, j]
                funcs[v][i, j] = make_1d_spline(z_col, y_col)
    return funcs

def interpolate_at_height(all_funcs, z_query):
    lon, lat = all_funcs[varnames[0]].shape
    out = {v: np.full((lon, lat), np.nan) for v in varnames}

    for i in range(lon):
        for j in range(lat):
            for v in varnames:
                out[v][i, j] = all_funcs[v][i, j](z_query)
    return out




# Test
z_t = ds_time["geopotential"].values / 9.80665   # (37, 240, 121)
z_q = 1000.0    # (37, 240, 121)

all_funcs = build_all_interp_funcs(z_t, ds_time, varnames)

result_1000m = interpolate_at_height(all_funcs, z_q)
u_1000  = result_1000m["u_component_of_wind"]
v_1000 = result_1000m["v_component_of_wind"]
temp_1000 = result_1000m["temperature"]
p_1000m = result_1000m["level"]
print(p_1000m)



