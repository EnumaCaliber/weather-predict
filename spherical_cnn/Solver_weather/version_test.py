import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os

# 读取 ERA5 数据
ds = xr.open_dataset("era5_20200601_12.nc")

# 常数
R = 287.0
Omega = 7.2921e-5

# 初始变量
u = np.clip(ds["10m_u_component_of_wind"].values, -20, 20)
v = np.clip(ds["10m_v_component_of_wind"].values, -20, 20)
T = ds["2m_temperature"].values
ps = ds["surface_pressure"].values
rho = ps / (R * T)

# 网格和 Coriolis
lon = ds["longitude"].values
lat = ds["latitude"].values
dx = 625471.4623756428
dy = 645647.9611619529
X, Y = np.meshgrid(lon, lat, indexing='ij')
lon2d, lat2d = np.meshgrid(lon, lat, indexing='ij')
f_cor = 2 * Omega * np.sin(np.deg2rad(lat2d))

# 差分算子（Neumann 边界）
def gradient_centered_neumann(field, dx, dy):
    padded = np.pad(field, 1, mode='edge')
    dfdx = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * dx)
    dfdy = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * dy)
    return dfdx, dfdy

def divergence_centered_neumann(fx, fy, dx, dy):
    fx_p = np.pad(fx, 1, mode='edge')
    fy_p = np.pad(fy, 1, mode='edge')
    return ((fx_p[2:, 1:-1] - fx_p[:-2, 1:-1]) / (2 * dx) +
            (fy_p[1:-1, 2:] - fy_p[1:-1, :-2]) / (2 * dy))

def laplacian_neumann(field, dx, dy):
    padded = np.pad(field, 1, mode='edge')
    return ((padded[2:, 1:-1] - 2 * padded[1:-1, 1:-1] + padded[:-2, 1:-1]) / dx**2 +
            (padded[1:-1, 2:] - 2 * padded[1:-1, 1:-1] + padded[1:-1, :-2]) / dy**2)

# 模拟参数
dt = 1.0
nu = 1e6
dt_max = 100.0
CFL = 0.5
tmax = 100

# 输出目录
output_dir = "wind_frames"
os.makedirs(output_dir, exist_ok=True)

# 时间推进 + 存图
for step in range(tmax):
    p = rho * R * T
    dpdx, dpdy = gradient_centered_neumann(p, dx, dy)
    dudx, dudy = gradient_centered_neumann(u, dx, dy)
    dvdx, dvdy = gradient_centered_neumann(v, dx, dy)

    adv_u = u * dudx + v * dudy
    adv_v = u * dvdx + v * dvdy
    lap_u = laplacian_neumann(u, dx, dy)
    lap_v = laplacian_neumann(v, dx, dy)

    coriolis_u = -f_cor * v
    coriolis_v = f_cor * u

    u_new = u + dt * (-adv_u - dpdx / rho + coriolis_u + nu * lap_u)
    v_new = v + dt * (-adv_v - dpdy / rho + coriolis_v + nu * lap_v)

    div_rho_v = divergence_centered_neumann(rho * u, rho * v, dx, dy)
    rho_new = rho - dt * div_rho_v

    max_u = np.max(np.sqrt(u**2 + v**2)) + 1e-5
    dt = min(CFL * min(dx, dy) / max_u, dt_max)

    if np.isnan(u_new).any() or np.isnan(rho_new).any():
        print(f"NaN at step {step}")
        break

    u, v, rho = u_new, v_new, rho_new

    # 每一步保存图片
    plt.figure(figsize=(8, 6))
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2], scale=500)
    plt.title(f"Wind Field at Step {step}")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wind_step_{step:03d}.png")
    plt.close()