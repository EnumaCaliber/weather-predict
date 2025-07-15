from utils.weather_data_process import WeatherDataProcess
import utils.picture_utils as picutils
from evaluation.weighted_acc_rmse import weighted_acc_torch
import torch
import numpy as np
from utils.weather_constant import WeatherConstant
file_path = "data/2021-01-01-24hours.nc"

WeatherDataProcess = WeatherDataProcess(file_path=file_path)
dx, dy = WeatherDataProcess.lat_lon_grid_deltas()
coriolis_force_u = WeatherDataProcess.get_u_coriolis_force()
coriolis_force_v = WeatherDataProcess.get_v_coriolis_force()
d_phi_dx = WeatherDataProcess.calculate_d_phi_dx_fdm(level=850)
d_phi_dy = WeatherDataProcess.calculate_d_phi_dy_fdm(level=850)
du_dt = d_phi_dx + coriolis_force_u
du_dv = d_phi_dy + coriolis_force_v
u = WeatherDataProcess.get_u_wind_speed(level=850,time_index= 0)
v = WeatherDataProcess.get_v_wind_speed(level=850,time_index= 0)
u_target = WeatherDataProcess.get_u_wind_speed(level=850, time_index=1)
v_target = WeatherDataProcess.get_v_wind_speed(level=850, time_index=1)

d_theta_dp = WeatherDataProcess.get_d_theta_d_p(level=850,time_index=0)



d_omega_dp = -WeatherDataProcess.get_du_dx(level=850,time_index=0) - WeatherDataProcess.get_du_dx(level=850,time_index=0)
dt_dx = WeatherDataProcess.get_dt_dx(level=850,time_index=0)
dt_dy = WeatherDataProcess.get_dt_dy(level=850,time_index=0)



gamma = WeatherDataProcess.get_dt_dz(level=850,time_index=0)
gamma_d= WeatherConstant.g / WeatherConstant.cp
omega_rho_g = WeatherDataProcess.get_w_rho_g(level=850,time_index=0)
ins_temp = omega_rho_g * (gamma - gamma_d)

alpha = WeatherDataProcess.get_alpha(level=850,time_index=0)

#TODO need interation

u_predict = u + du_dt * 3600
v_predict = v + du_dt * 3600
wind_speed_pred = np.sqrt(u_predict**2 + v_predict**2)


# acc_result = weighted_acc_torch(torch.from_numpy(u_target).float().unsqueeze(0).unsqueeze(0),
#                                 torch.from_numpy(u_predict).float().unsqueeze(0).unsqueeze(0))

lon = WeatherDataProcess.get_longitude()
lat = WeatherDataProcess.get_latitude()

picutils.draw_clean(pic=wind_speed_pred, lon=lon, lat=lat, scale=1, title="wind_speed_pred")
picutils.draw_clean(pic=u_predict, lon=lon, lat=lat, scale=1, title="u_predict")
picutils.draw_clean(pic=v_predict, lon=lon, lat=lat, scale=1, title="v_predict")

