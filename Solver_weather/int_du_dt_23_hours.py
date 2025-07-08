import xarray as xr
import torch
from Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch
from Solver_weather.weather_util import get_point_parameters
from pic_util import *
import metpy.calc as mpcalc

file_path = "era5_day_2021-01-01.nc"
ds = xr.open_dataset(file_path)

time = ds.time.values
level = 100
diffusion_coefficient_flat = 10e3
diffusion_coefficient_vertical = 1
residuals = []
acc_list = []

du_dt_list = []
u_curr_list = []
u_next_list = []
total_time = 24
for time_index in range(0, total_time, 1):
    time_curr = ds.time.values[time_index]
    time_next = ds.time.values[time_index]
    ds_curr = ds.sel(time=time_curr)
    ds_next = ds.sel(time=time_next)
    util_curr = get_point_parameters(ds_curr)
    util_next = get_point_parameters(ds_next)
    ds_metpy_cur = ds.metpy.parse_cf(ds).sel(time=time_curr)
    ##########u_advection##########
    # duu_dx = util_curr.d_x(level=level, wind_type="uu")
    # duv_dy = util_curr.d_y(level=level, wind_type="uv")
    # duw_dz = util_curr.d_z(level=[level, level + 50], wind_type="uw")
    # u_advection = -(duu_dx + duv_dy + duw_dz)
    u = ds_metpy_cur.sel(level=level)["u_component_of_wind"]
    v = ds_metpy_cur.sel(level=level)["v_component_of_wind"]
    u_advection_metpy = mpcalc.advection(u, u=u, v=v, x_dim=-2, y_dim=-1, vertical_dim=-3).metpy.magnitude
    u_advection = u_advection_metpy.copy()
    u_advection[0, :] /= 10000
    u_advection[-1, :] /= 10000

    # 左右列（去掉已经除过的角点）
    u_advection[1:-1, 0] /= 10000
    u_advection[1:-1, -1] /= 10000
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
    du_ddz = util_curr.dd_z(level=[level, level + 50], du1=du_dz, du2=du_dz_2)
    # du_ddz = util_curr.dd_z_center(levels=[level - 50, level, level + 50])
    diffusion = diffusion_coefficient_flat * (du_ddx + du_ddy) + diffusion_coefficient_vertical * du_ddz
    ##########diffusion##########

    ##########dudt##########
    du_dt_exp = u_advection + PGF + coriolis + diffusion
    ##########dudt##########
    u_curr = util_curr.get_wind_u(level=level)
    u_next = util_next.get_wind_u(level=level)
    du_dt_list.append(du_dt_exp)
    u_curr_list.append(u_curr)
    u_next_list.append(u_next)

t_old = np.arange(total_time)  # 每小时一个点：0~23
du_dt_array = np.stack(du_dt_list)  # [24, lat, lon]
u_array = np.stack(u_curr_list)  # [24, lat, lon]

lon, lat = u_array.shape[1:]
u_interp = np.zeros_like(u_array)

u_reconstructed = np.zeros_like(du_dt_array)
u_reconstructed[0] = u_curr_list[0]
u_init = u_curr_list[0]
# for i in range(lon):
#     for j in range(lat):
#         u_series = u_array[:, i, j]
#         du_series = du_dt_array[:, i, j]
#
#         if np.any(np.isnan(u_series)) or np.any(np.isnan(du_series)):
#             continue
#
#         spline = CubicHermiteSpline(t_old, du_series, np.zeros_like(du_series))
#
#         for k in range(1, len(t_old)):
#             integral = spline.integrate(t_old[k - 1], t_old[k]) * 3600
#             u_reconstructed[k, i, j] = u_reconstructed[k - 1, i, j] + integral


for t in range(1, total_time):
    u_reconstructed[t] = u_reconstructed[t - 1] + du_dt_array[t - 1] * 3600

u_next_np = np.stack(u_next_list[:len(u_reconstructed)])
acc_list = []
rmse_list = []
u_pred = u_reconstructed

lat_vals = ds["latitude"].values
lat_mask = (lat_vals >= -50) & (lat_vals <= 50)
lat_indices = np.where(lat_mask)[0]


draw(u_reconstructed[-1],lat=ds_curr["latitude"],lon=ds_curr["longitude"],scale=1)
draw(u_next_np[-1],lat=ds_curr["latitude"],lon=ds_curr["longitude"],scale=1)

for t in range(0, total_time, 1):
    pred = torch.from_numpy(u_pred[t][None, None]).float()
    true = torch.from_numpy(u_next_np[t][None, None]).float()
    pred_eq = pred[:, :, :, lat_indices]  # 只取赤道区域
    true_eq = true[:, :, :, lat_indices]

    acc_t = weighted_acc_torch_channels(pred_eq, true_eq).item()
    rmse_t = weighted_rmse_torch(pred_eq, true_eq).item()

    acc_list.append(acc_t)
    rmse_list.append(rmse_t)

time_hours = np.arange(0, total_time)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time_hours, acc_list, marker='o', linestyle='-', color='teal')
plt.title("ACC per Hour")
plt.xlabel("Hour")
plt.ylabel("ACC")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(time_hours, rmse_list, marker='o', linestyle='-', color='darkorange')
plt.title("RMSE per Hour")
plt.xlabel("Hour")
plt.ylabel("RMSE")
plt.grid(True)




plt.tight_layout()
plt.show()


print(acc_list)
print(rmse_list)