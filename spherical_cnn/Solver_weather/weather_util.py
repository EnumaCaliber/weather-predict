import xarray as xr
import numpy as np


# constant define
R = 287.0
P = 85000


class get_point_parameters:
    def __init__(self, file_path):
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path)

        self.cos_alpha = 1
        self.sin_alpha = 0
        self.omega = 7.2921e-5
        self.g = 9.80665
    def get_rho(self,level):
        ds = self.ds.sel(level=level)
        q = ds["specific_humidity"].values # specific_humidity
        T = ds["temperature"].values

        rho = P / (R * T * (1 + 0.61 * q))
        return rho


    def get_t_virtual(self,level):
        ds = self.ds.sel(level=level)
        q = ds["specific_humidity"].values # specific_humidity
        T = ds["temperature"].values
        T_virtual = T * (1 + 0.61 * q)
        return T_virtual

    def get_u_coriolis_force(self,level):
        ds = self.ds.sel(level=level)

        lat = ds["latitude"].values
        v = ds["v_component_of_wind"].values
        w = ds["vertical_velocity"].values

        lat = np.tile(lat[:, np.newaxis], (1, 32))

        fv = 2 * self.omega * np.sin(lat) * v
        ew = 2 * self.omega * np.cos(lat) * w * self.cos_alpha
        fw = 2 * self.omega * np.sin(lat) * w * self.sin_alpha
        return (fv + ew - fw)


    def get_v_coriolis(self,level):
        ds = self.ds.sel(level=level)
        lat = ds["latitude"].values
        lat = np.tile(lat[:, np.newaxis], (1, 32))

        u = ds["u_component_of_wind"].values
        w = ds["vertical_velocity"].values
        fu = 2 * self.omega * np.sin(lat) * u
        ew = 2 * self.omega * np.cos(lat) * w * self.sin_alpha
        fw = 2 * self.omega * np.sin(lat) * w * self.cos_alpha
        return (- fu + ew + fw)

    def get_w_coriolis(self,level):
        ds = self.ds.sel(level=level)
        u = ds["u_component_of_wind"].values
        v = ds["v_component_of_wind"].values
        lat = ds["latitude"].values
        lat = np.tile(lat[:, np.newaxis], (1, 32))
        coriolis_force = 2 * self.omega * np.cos(lat) * (u * self.cos_alpha + v * self.sin_alpha)
        return (- coriolis_force)

    def get_high_meter(self,level):
        ds = self.ds.sel(level=level)
        geopotential = ds["geopotential"].values
        high = geopotential / self.g
        return high


    def get_high_diff(self,level1,level2):
        ds_level1 = self.ds.sel(level=level1)
        ds_level2 = self.ds.sel(level=level2)
        geopotential1 = ds_level1["geopotential"].values
        geopotential2 = ds_level2["geopotential"].values

        high_diff = abs((geopotential1 - geopotential2) / self.g)
        return high_diff


    def get_lat_distance(self,level):
        EARTH_RADIUS_M = 6370000
        ds = self.ds.sel(level=level)
        # meter
        lat = ds["latitude"].values
        delta_lat = lat[1] - lat[0]
        distance = delta_lat * 2 * np.pi *  EARTH_RADIUS_M / 360
        return distance

    def get_lon_distance(self,level):
        EARTH_RADIUS_M = 6370000
        ds = self.ds.sel(level=level)
        # meter
        lon = ds["longitude"].values
        delta_lon = lon[1] - lon[0]
        distance = delta_lon * 2 * np.pi *  EARTH_RADIUS_M / 360
        return distance

    def get_wind_u(self,level):
        ds = self.ds.sel(level=level)
        u = ds["u_component_of_wind"].values
        return u

    def get_wind_v(self,level):
        ds = self.ds.sel(level=level)
        v = ds["v_component_of_wind"].values
        return v

    def get_wind_w(self,level):
        ds = self.ds.sel(level=level)
        w = ds["vertical_velocity"].values
        return w

    def get_level(self):
        level = self.ds["level"].values
        return level






