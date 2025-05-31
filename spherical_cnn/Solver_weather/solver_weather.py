# === 修改后的 Dedalus 模拟代码 ===
import os
import numpy as np
import xarray as xr
import dedalus.public as d3
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from dedalus.extras import flow_tools

# ------------------ 基础设置 ------------------
os.environ["OMP_NUM_THREADS"] = "1"  # 限制线程数以提升稳定性

# ------------------ 时间设置 ------------------
hour = 1
second = hour / 3600
initial_dt = 1e-3  # ✅ 初始时间步设为小值（建议结合 CFL）
stop_sim_time = 1 * hour

# ------------------ 读取 ERA5 数据 ------------------
ds = xr.open_dataset("era5_20200601_12.nc")
u10 = ds["10m_u_component_of_wind"].values[::-1, :]
v10 = ds["10m_v_component_of_wind"].values[::-1, :]
t2m = ds["2m_temperature"].values[::-1, :]
p_surface = ds["surface_pressure"].values[::-1, :]

Ny, Nx = u10.shape
assert u10.shape == t2m.shape == p_surface.shape, "ERA5 变量尺寸不一致"

# ------------------ 参数设置 ------------------
Lx, Ly = 1.0, 1.0
R = 287.0
cp = 1004.0
nu = 1e-2        # ✅ 增大粘性，避免高频能量堆积
kappa = 1e-2     # ✅ 增大热扩散系数
f = 1e-4

# ------------------ 初始化密度 ------------------
rho_initial = p_surface / (R * t2m)

# ------------------ 建立网格 ------------------
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=np.float64)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx))
ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly))

# ------------------ 定义变量 ------------------
u1 = dist.Field(name='u1', bases=(xbasis, ybasis))
u2 = dist.Field(name='u2', bases=(xbasis, ybasis))
T  = dist.Field(name='T',  bases=(xbasis, ybasis))
rho = dist.Field(name='rho', bases=(xbasis, ybasis))
p   = dist.Field(name='p', bases=(xbasis, ybasis))

# ------------------ 初始条件赋值（滤波） ------------------
# ✅ 对风场做高斯滤波，消除ERA5数据中的高频成分
u1['g'] = gaussian_filter(u10.T, sigma=3)
u2['g'] = gaussian_filter(v10.T, sigma=3)
T['g']  = gaussian_filter(t2m.T, sigma=3)
rho['g'] = p_surface.T / (R * T['g'])  # ✅ 确保状态方程满足
p['g'] = p_surface.T

print(f"initial max and min: u1=({u1['g'].max():.2f},{u1['g'].min():.2f}) "
      f"u2=({u2['g'].max():.2f},{u2['g'].min():.2f}) "
      f"T=({T['g'].max():.2f},{T['g'].min():.2f}) "
      f"rho=({rho['g'].max():.2f},{rho['g'].min():.2f}) "
      f"p=({p['g'].max():.2f},{p['g'].min():.2f})")

# ------------------ 建立 PDE ------------------
problem = d3.IVP([u1, u2], namespace=locals())
dx = lambda x: d3.Differentiate(x, coords['x'])
dy = lambda x: d3.Differentiate(x, coords['y'])
lap = lambda x: dx(dx(x)) + dy(dy(x))
eps = 1e-2  #

problem.add_equation("dt(u1) = - u1*dx(u1) - u2*dy(u1) + f*u2 - dx(p)/(rho+eps) + nu*lap(u1)")
problem.add_equation("dt(u2) = - u1*dx(u2) - u2*dy(u2) - f*u1 - dy(p)/(rho+eps) + nu*lap(u2)")
problem.add_equation("dt(T)  = - u1*dx(T)  - u2*dy(T) + kappa*lap(T)")
problem.add_equation("dt(rho) = - u1*dx(rho) - rho*dx(u1) - u2*dy(rho) - rho*dy(u2)")
problem.add_equation("p = rho*R*T")

# ------------------ 构建求解器 ------------------
solver = problem.build_solver(d3.SBDF2)
solver.stop_sim_time = stop_sim_time
solver.dt = initial_dt  # ✅ 设置初始时间步

# ------------------ 监控设置 ------------------
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property(u1*u1 + u2*u2, name='U2')

# ------------------ 时间推进 ------------------
print("solving...")
snapshots = []

try:
    while solver.proceed:
        solver.step(solver.dt)

        if solver.iteration % 2 == 0:
            u1_g = u1['g']
            maxval, minval = np.nanmax(u1_g), np.nanmin(u1_g)
            print(f"Iter {solver.iteration:04d} | t = {solver.sim_time:.4f} | max(u1) = {maxval:.3e}, min = {minval:.3e} | KE = {flow.max('U2'):.3e}")
            if np.isnan(maxval) or np.isnan(minval):
                print("检测到 NaN，停止模拟")
                break
            snapshots.append(u1_g.copy())
except Exception as e:
    print(f"模拟异常终止: {e}")

# ------------------ 可视化 ------------------
if snapshots:
    plt.figure(figsize=(6, 4))
    plt.imshow(snapshots[-1], origin='lower', cmap='RdBu', extent=(0, Lx, 0, Ly))
    plt.colorbar(label="u1 (m/s)")
    plt.title("Final Zonal Wind (u1) Snapshot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()
else:
    print("No snapshots collected.")
