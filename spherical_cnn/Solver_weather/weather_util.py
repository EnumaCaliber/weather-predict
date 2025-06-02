from sympy.simplify import fu
from wrf import getvar, interplevel
import xarray as xr
import numpy as np
# constant define
R = 287.0
P = 85000


class get_point_parameters:
    def __init__(self, file_path, level):
        self.file_path = file_path
        self.ds = xr.open_dataset(file_path).sel(level=level)
        self.level = level
        self.cos_alpha = 1
        self.sin_alpha = 0
        self.omega = 7.2921e-5
    def get_rho(self):
        ds = self.ds
        q = ds["specific_humidity"].values # specific_humidity
        T = ds["temperature"].values

        rho = P / (R * T * (1 + 0.61 * q))
        return rho


    def get_t_virtual(self):
        ds = self.ds
        q = ds["specific_humidity"].values # specific_humidity
        T = ds["temperature"].values
        T_virtual = T * (1 + 0.61 * q)
        return T_virtual

    def get_u_coriolis_force(self):
        ds = self.ds

        lat = ds["latitude"].values
        v = ds["v_component_of_wind"].values
        w = ds["vertical_velocity"].values

        lat = np.tile(lat[:, np.newaxis], (1, 32))

        fv = 2 * self.omega * np.sin(lat) * v
        ew = 2 * self.omega * np.cos(lat) * w * self.cos_alpha
        fw = 2 * self.omega * np.sin(lat) * w * self.sin_alpha
        return (fv + ew - fw)


    def get_v_coriolis(self):
        lat = self.ds["latitude"].values
        lat = np.tile(lat[:, np.newaxis], (1, 32))

        ds = self.ds
        u = ds["u_component_of_wind"].values
        w = ds["vertical_velocity"].values
        fu = 2 * self.omega * np.sin(lat) * u
        ew = 2 * self.omega * np.cos(lat) * w * self.sin_alpha
        fw = 2 * self.omega * np.sin(lat) * w * self.cos_alpha
        return (- fu + ew + fw)

    def get_w_coriolis(self):
        u = self.ds["u_component_of_wind"].values
        v = self.ds["v_component_of_wind"].values
        lat = self.ds["latitude"].values
        lat = np.tile(lat[:, np.newaxis], (1, 32))
        coriolis_force = 2 * self.omega * np.cos(lat) * (u * self.cos_alpha + v * self.sin_alpha)
        return (- coriolis_force)

