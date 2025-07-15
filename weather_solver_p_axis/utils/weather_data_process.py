import xarray as xr
import numpy as np
from .weather_constant import WeatherConstant
import metpy.calc as mpcalc
from metpy.units import units


def fdm_2_order(value, grid):
    d_value = np.zeros_like(value)
    grid_ = grid
    value_ = value
    d_value[:, 2:-2] = (- value_[:, 4:] + 8 * value_[:, 3:-1] - 8 * value_[:, 1:-3] + value_[:, 0:-4]) / (
                12 * grid_[:, 2:-2])
    d_value[:, 1] = (value_[:, 2] - value_[:, 0]) / (2 * grid_[:, 1])
    d_value[:, 0] = (value_[:, 1] - value_[:, 0]) / grid_[:, 0]
    d_value[:, -2] = (value_[:, -1] - value_[:, -3]) / (2 * grid_[:, -2])
    d_value[:, -1] = (value_[:, -1] - value_[:, -2]) / grid_[:, -1]
    return d_value


class WeatherDataProcess:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path)
        self.metpy_ds = self.ds.metpy.parse_cf(self.ds)
        self.times = self.ds.time.values

    def lat_lon_grid_deltas(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        x_lon, y_lat = np.meshgrid(lon, lat)
        d_lon_y, d_lon_x = np.gradient(x_lon)
        d_lat_y, d_lat_x = np.gradient(y_lat)
        high_lat_mask = np.abs(y_lat) > 89.0
        dx = WeatherConstant.re * np.cos(y_lat * WeatherConstant.pi / 180) * d_lon_x * WeatherConstant.pi / 180
        dy = WeatherConstant.re * d_lat_y * WeatherConstant.pi / 180
        dx[high_lat_mask] = dy[high_lat_mask]
        return dx.T, dy.T

    def get_u_coriolis_force(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        ds_metpy = self.metpy_ds.sel(level=850, time=self.times[0])
        v = ds["v_component_of_wind"].values
        lon_size = ds["longitude"].values.size
        lat = ds_metpy['latitude'].values * units.degrees
        coriolis = mpcalc.coriolis_parameter(lat).magnitude
        coriolis = np.tile(coriolis[np.newaxis, :], (lon_size, 1))
        return coriolis * v

    def get_v_coriolis_force(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        ds_metpy = self.metpy_ds.sel(level=850, time=self.times[0])
        u = ds["u_component_of_wind"].values
        lon_size = ds["longitude"].values.size
        lat = ds_metpy['latitude'].values * units.degrees
        coriolis = mpcalc.coriolis_parameter(lat).magnitude
        coriolis = np.tile(coriolis[np.newaxis, :], (lon_size, 1))
        return -coriolis * u

    def calculate_d_phi_dx_fdm(self, level):
        ds = self.ds.sel(level=level, time=self.times[0])
        phi = ds['geopotential'].values
        dx, _ = self.lat_lon_grid_deltas()
        d_phi_dx = fdm_2_order(phi,dx)
        return -d_phi_dx

    def calculate_d_phi_dy_fdm(self, level):
        ds = self.ds.sel(level=level, time=self.times[0])
        phi = ds['geopotential'].values
        _, dy = self.lat_lon_grid_deltas()
        d_phi_dy = fdm_2_order(phi, dy)
        return -d_phi_dy

    def get_longitude(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        lon = ds["longitude"].values
        return lon

    def get_latitude(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        lat = ds["latitude"].values
        return lat

    def get_u_wind_speed(self, level, time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        u = ds["u_component_of_wind"].values
        return u

    def get_v_wind_speed(self, level, time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        v = ds["v_component_of_wind"].values
        return v

    def get_d_theta_d_p(self, level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        temperature = ds["temperature"].values
        r_const = WeatherConstant.R
        pressure = level * 100
        return -r_const*temperature/pressure

    def get_du_dx(self, level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        u = ds["u_component_of_wind"].values
        dx, _ = self.lat_lon_grid_deltas()
        du_dx = fdm_2_order(u,dx)
        return du_dx

    def get_dv_dy(self, level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        v = ds["v_component_of_wind"].values
        _, dy = self.lat_lon_grid_deltas()
        dv_dy = fdm_2_order(v,dy)
        return dv_dy

    def get_dt_dx(self, level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        temperature = ds["temperature"].values
        dx,_ = self.lat_lon_grid_deltas()
        dt_dx = fdm_2_order(temperature,dx)
        return dt_dx

    def get_dt_dy(self, level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        temperature = ds["temperature"].values
        _,dy = self.lat_lon_grid_deltas()
        dt_dy = fdm_2_order(temperature,dy)
        return dt_dy

    def get_density(self, level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        p = level * 100
        r = WeatherConstant.R
        t = ds["temperature"].values
        return p/(r*t)


    def get_dt_dz(self, level,time_index):
        ds_low = self.ds.sel(level=level + 50, time=self.times[time_index])
        ds_up = self.ds.sel(level=level - 50, time=self.times[time_index])
        z_up = ds_up["geopotential"]/WeatherConstant.g
        z_low = ds_low["geopotential"]/WeatherConstant.g
        t_low = ds_low["temperature"]
        t_up = ds_up["temperature"]

        dt_dz = (t_up - t_low) / (z_up - z_low)
        return dt_dz

    def get_w_rho_g(self,level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        omega = ds["vertical_velocity"].values
        rho = self.get_density(level,time_index)
        g = WeatherConstant.g
        omega_rho_g = omega / (rho*g)
        return omega_rho_g


    def get_alpha(self,level,time_index):
        ds = self.ds.sel(level=level, time=self.times[time_index])
        t = ds["temperature"].values
        p = level * 100
        r = WeatherConstant.R
        alpha = r*t/p
        return alpha