import xarray as xr
import numpy as np
from .weather_constant import WeatherConstant
import metpy.calc as mpcalc
from metpy.units import units


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

    def calculate_d_phi_dx(self, level):
        ds = self.ds.sel(level=level, time=self.times[0])
        phi = ds['geopotential'].values
        dx, _ = self.lat_lon_grid_deltas()
        d_phi_dx, _ = np.gradient(phi)
        d_phi_dx = d_phi_dx / dx  # ∂Φ/∂x
        return -d_phi_dx



    def calculate_d_phi_dx_fdm(self, level):
        ds = self.ds.sel(level=level, time=self.times[0])
        phi = ds['geopotential'].values
        d_phi_dx = np.zeros_like(phi)
        dx, _ = self.lat_lon_grid_deltas()
        d_phi_dx[2:-2, :] = (-phi[4:, :] + 8 * phi[3:-1, :] - 8 * phi[1:-3, :] + phi[0:-4, :]) / (12 * dx[2:-2, :])
        d_phi_dx[1, :] = (phi[2, :] - phi[0, :]) / (2 * dx[1, :])
        d_phi_dx[0, :] = (phi[1, :] - phi[0, :]) / dx[0, :]
        d_phi_dx[-2, :] = (phi[-1, :] - phi[-3, :]) / (2 * dx[-2, :])
        d_phi_dx[-1, :] = (phi[-1, :] - phi[-2, :]) / dx[-1, :]
        return d_phi_dx

    def get_longitude(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        lon = ds["longitude"].values
        return lon

    def get_latitude(self):
        ds = self.ds.sel(level=850, time=self.times[0])
        lat = ds["latitude"].values
        return lat
