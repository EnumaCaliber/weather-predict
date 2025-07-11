from utils.weather_data_process import WeatherDataProcess
import utils.picture_utils as picutils

file_path = "data/2021-01-01-24hours.nc"

WeatherDataProcess = WeatherDataProcess(file_path=file_path)
dx, dy = WeatherDataProcess.lat_lon_grid_deltas()
coriolis_force_u = WeatherDataProcess.get_u_coriolis_force()
d_phi_dx = WeatherDataProcess.calculate_d_phi_dx_fdm(level=850)
du_dt = d_phi_dx + coriolis_force_u
lon = WeatherDataProcess.get_longitude()
lat = WeatherDataProcess.get_latitude()
picutils.draw_clean(pic=du_dt, lon=lon, lat=lat, scale=1)
