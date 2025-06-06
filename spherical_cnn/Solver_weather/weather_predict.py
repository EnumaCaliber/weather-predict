

import numpy as np


from spherical_cnn.Solver_weather.weather_util import get_point_parameters
from pic_util import *


file_path_1 = "era5_20200601_12_1.nc"

diffusion_coefficient_flat = 1000
diffusion_coefficient_vertical = 5


util = get_point_parameters(file_path_1)

lon = util.get_lon(level=850)
lat = util.get_lat(level=850)
u = util.get_wind_u(level=850)

duu_dx = util.d_x(level=850, wind_type ="uu")
duv_dy = util.d_y(level=850, wind_type ="uv")
duw_dz = util.d_z(level=[850,925], wind_type ="uw")


du_dx = util.d_x(level=850, wind_type ="u")
du_dy = util.d_y(level=850, wind_type ="u")
du_dz = util.d_z(level=[850,925], wind_type ="u")



du_dz_2 = util.d_z(level=[925,1000], wind_type ="u")
du_ddz = util.dd_z(level=[850,925],du1=du_dz,du2=du_dz_2)

v = util.get_wind_v(level=850)
dv_dx = util.d_x(level=850, wind_type ="v")
dv_dy = util.d_y(level=850, wind_type ="v")
dv_dz = util.d_z(level=[850,925], wind_type ="v")




w = util.get_wind_v(level=850)
dw_dx = util.d_x(level=850, wind_type ="v")
dw_dy = util.d_y(level=850, wind_type ="v")
dw_dz = util.d_z(level=[850,925], wind_type ="v")

rho = util.get_rho(level=850)

p_true = util.get_true_pressure(level=850)



u_advection = -(duu_dx + duv_dy + duw_dz)
dp_dx = util.d_x(level=850, wind_type ="p")
p_gradient = - util.get_geopotential_dx(level=850) * 9.8


rho = util.get_rho(level=850)


du_ddx = util.dd_x(dx = du_dx,level=850)
# 存在条纹
du_ddy = util.dd_y(dy = du_dy,level=850)

# function construction
PGF = -( 1 / rho) * dp_dx
# 存在条纹
coriolis = util.get_u_coriolis_force(level=850)
diffusion = diffusion_coefficient_flat * (du_ddx + du_ddy) +  diffusion_coefficient_vertical*du_ddz




# total = u_advection + PGF + coriolis + diffusion
total = u_advection  - 9.8 * p_gradient + coriolis + diffusion
total_2 = u_advection  + PGF + coriolis + diffusion
file_path_2 = "era5_20200601_12_2.nc"
util_2 = get_point_parameters(file_path_2)
u_t2 = util_2.get_wind_u(level=850)

du_dt = ((u + u_t2)/2 - u) / (1800)



draw(total, lon=lon, lat=lat,scale=1,title="total")
draw(total_2, lon=lon, lat=lat,scale=1,title="total_2")
draw(du_dt, lon=lon, lat=lat,scale=1,title="du_dt")

