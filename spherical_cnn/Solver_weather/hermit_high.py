import xarray as xr
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from pic_util import *
from spherical_cnn.Solver_weather.random_sample_test_du_dt import level

file_path = "era5_day_2021-01-01.nc"

ds = xr.open_dataset(file_path)
time_curr = ds.time.values[0]
time_next = ds.time.values[1]
ds_time = ds.sel(time=time_curr)
ds_next_time = ds.sel(time=time_next)

varnames = ["u_component_of_wind", "v_component_of_wind",
            "temperature", "specific_humidity","vertical_velocity",
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

def differentiate_at_height(all_funcs, z_query, n):
    lon, lat = all_funcs[varnames[0]].shape
    out = {v: np.full((lon, lat), np.nan) for v in varnames}

    for i in range(lon):
        for j in range(lat):
            for v in varnames:
                try:
                    out[v][i, j] = all_funcs[v][i, j].derivative(nu = n)(z_query)
                except:
                    out[v][i, j] = np.nan
    return out


# Test
z_t = ds_time["geopotential"].values / 9.80665   # (37, 240, 121)
z_q = 1000.0    # (37, 240, 121)

all_funcs = build_all_interp_funcs(z_t, ds_time, varnames)
result_1000m = interpolate_at_height(all_funcs, z_q)
grad_1000m = differentiate_at_height(all_funcs, 1000.0,n = 1)
grad_2_1000m = differentiate_at_height(all_funcs, 1000.0,n = 2)

u_1000  = result_1000m["u_component_of_wind"]
v_1000 = result_1000m["v_component_of_wind"]
w_1000 = result_1000m["vertical_velocity"]
temp_1000 = result_1000m["temperature"]
p_1000m = result_1000m["level"]
sp_1000m = result_1000m["specific_humidity"]

du_dz_1000 = grad_1000m["u_component_of_wind"]
dw_dz_1000 = grad_1000m["vertical_velocity"]

du_ddz_100 = grad_2_1000m["u_component_of_wind"]
EARTH_RADIUS_M = 6370000
# meterc
lon = ds_time["longitude"].values
delta_lon = lon[1] - lon[0]
lon_dis = abs(delta_lon * 2 * np.pi * EARTH_RADIUS_M / 360)
lat = ds_time["latitude"].values
delta_lat = lat[1] - lat[0]
lat_dis = delta_lat * 2 * np.pi * EARTH_RADIUS_M / 360


def compute_du_dt(u, v, w, temp, sp, p, z, lon, lat):    # meter

    ##########u_advection##########
    duu_dx = (np.roll(u_1000 * u_1000, -1, axis=0) - np.roll(u_1000 * u_1000, 1, axis=0)) / (2 * lon_dis)

    duv_dy = np.zeros_like(v_1000)
    duv_dy[:, 1:-1] = ((u_1000 * v_1000)[:, 2:] - (u_1000 * v_1000)[:, :-2]) / (2 * lat_dis)
    # 前向差分（南边界）
    duv_dy[:, 0] = ((u_1000 * v_1000)[:, 1] - (u_1000 * v_1000)[:, 0]) / lat_dis
    # 后向差分（北边界）
    duv_dy[:, -1] = ((u_1000 * v_1000)[:, -1] - (u_1000 * v_1000)[:, -2]) / lat_dis

    duw_dz = u_1000* dw_dz_1000 + w_1000 * du_dz_1000
    u_advection = - (duu_dx + duv_dy + duw_dz)
    ##########u_advection##########


    dp_dx = (np.roll(p_1000m, -1, axis=0) - np.roll(p_1000m, 1, axis=0)) / (2 * lon_dis)
    rho = p_1000m / (287 * temp_1000 * (1 + 0.61 * sp_1000m))
    PGF = -( 1 / rho) * dp_dx

    du_dx = (np.roll(u_1000, -1, axis=0) - np.roll(u_1000, 1, axis=0)) / (2 * lon_dis)
    du_dy = np.zeros_like(u_1000)
    du_dy[:, 1:-1] = (u_1000[:, 2:] - u_1000[:, :-2]) / (2 * lat_dis)
    # 前向差分（南边界）
    du_dy[:, 0] = (u_1000[:, 1] - u_1000[:, 0]) / lat_dis
    # 后向差分（北边界）
    du_dy[:, -1] = (u_1000[:, -1] - u_1000[:, -2]) / lat_dis

    du_dz = du_dz_1000
    du_ddz = du_ddz_100
    du_ddx = (np.roll(du_dx, -1, axis=0) - np.roll(du_dx, 1, axis=0)) / (2 * lon_dis)
    du_ddy = np.zeros_like(u_1000)
    du_ddy[:, 1:-1] = (du_dy[:, 2:] - du_dy[:, :-2]) / (2 * lat_dis)
    # 前向差分（南边界）
    du_dy[:, 0] = (du_dy[:, 1] - du_dy[:, 0]) / lat_dis
    # 后向差分（北边界）
    du_dy[:, -1] = (du_dy[:, -1] - du_dy[:, -2]) / lat_dis
    diffusion = 10e-5 * (du_ddx + du_ddy) + 1 * du_ddz


    lon_size = ds_time["longitude"].values.size
    v = v_1000

    ##########################
    w = w_1000
    ##########################
    cos_alpha = 1
    sin_alpha = 0
    omega = 7.2921e-5
    lat = np.tile(lat[np.newaxis, :], (lon_size, 1))
    lat_rad = np.deg2rad(lat)
    fv = 2 * omega * np.sin(lat_rad) * v_1000
    ew = 2 * omega * np.cos(lat_rad) * w_1000 * cos_alpha
    fw = 2 * omega * np.sin(lat_rad) * w_1000 * sin_alpha
    u_coriolis_force = fv + ew + fw
    du_dt_exp = u_advection + PGF + u_coriolis_force
    return du_dt_exp

u0 = u_1000
v = v_1000
w = w_1000
temp = temp_1000
sp = sp_1000m
p = p_1000m
z = z_q
dt = 3600

k1 = compute_du_dt(u0, v, w, temp, sp, p, z, lon, lat)
k2 = compute_du_dt(u0 + 0.5 * dt * k1, v, w, temp, sp, p, z, lon, lat)
k3 = compute_du_dt(u0 + 0.5 * dt * k2, v, w, temp, sp, p, z, lon, lat)
k4 = compute_du_dt(u0 + dt * k3, v, w, temp, sp, p, z, lon, lat)

u_rk4 = u0 + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

u_next_true = ds_next_time.sel(level=850)["u_component_of_wind"].values
draw(u_rk4,lon,lat,scale=1)
draw(u_next_true,lon,lat,scale=1)
draw(u_1000,lon,lat,scale=1)