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
        lon = ds["longitude"].values.size
        lat = ds["latitude"].values
        v = ds["v_component_of_wind"].values
        w = ds["vertical_velocity"].values

        lat = np.tile(lat[np.newaxis, :], (lon, 1))

        fv = 2 * self.omega * np.sin(lat) * v
        ew = 2 * self.omega * np.cos(lat) * w * self.cos_alpha
        fw = 2 * self.omega * np.sin(lat) * w * self.sin_alpha
        return (fv + ew - fw)


    def get_v_coriolis(self,level):
        ds = self.ds.sel(level=level)
        lon = ds["longitude"].values.size
        lat = ds["latitude"].values
        lat = np.tile(lat[np.newaxis, :], (lon, 1))

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

    def get_lon(self,level):
        ds = self.ds.sel(level=level)
        lon = ds["longitude"].values
        return lon

    def get_lat(self,level):
        ds = self.ds.sel(level=level)
        lat = ds["latitude"].values
        return lat


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

    # lon
    def d_x(self, level, wind_type):

        lon_dis = self.get_lon_distance(level=level)
        if wind_type == "u":
            wind = self.get_wind_u(level=level)
        elif wind_type == "v":
            wind = self.get_wind_v(level=level)
        else:
            wind = self.get_wind_w(level=level)
        dx = np.zeros_like(wind)
        dx = (np.roll(wind, -1, axis=0) - np.roll(wind, 1, axis=0)) / (2 * lon_dis)
        return dx


    # lat
    def d_y(self, level, wind_type):
        lat_dis = self.get_lat_distance(level=level)
        if wind_type == "u":
            wind = self.get_wind_u(level=level)
        elif wind_type == "v":
            wind = self.get_wind_v(level=level)
        elif wind_type == "w":
            wind = self.get_wind_w(level=level)
        else:
            wind = self.get_wind_w(level=level)
        dy = np.zeros_like(wind)
        # 中心差分（内部点）
        dy[:, 1:-1] = (wind[:, 2:] - wind[:, :-2]) / (2 * lat_dis)
        # 前向差分（南边界）
        dy[:, 0] = (wind[:, 1] - wind[:, 0]) / lat_dis
        # 后向差分（北边界）
        dy[:, -1] = (wind[:, -1] - wind[:, -2]) / lat_dis
        return dy

    # this level should be two level [level1, level2]
    def d_z(self, level, wind_type):
        level.sort()
        level1 = level[0]
        level2 = level[1]
        high_diff = self.get_high_diff(level1=level1,level2=level2)
        if wind_type == "u":
            wind1 = self.get_wind_u(level=level1)
            wind2 = self.get_wind_u(level=level2)
        elif wind_type == "v":
            wind1 = self.get_wind_v(level=level1)
            wind2 = self.get_wind_v(level=level2)
        else:
            wind1 = self.get_wind_w(level=level1)
            wind2 = self.get_wind_w(level=level2)
        dz = (wind1 - wind2) / (high_diff)
        return dz


    def dd_x(self, dx,level):
        lon_dis = self.get_lon_distance(level=level)
        ddx = np.zeros_like(dx)
        ddx = (np.roll(dx, -1, axis=0) - np.roll(dx, 1, axis=0)) / (2 * lon_dis)
        return ddx

    def dd_y(self, dy,level):
        lat_dis = self.get_lat_distance(level=level)
        ddy = np.zeros_like(dy)
        ddy[:, 1:-1] = (dy[:, 2:] - dy[:, :-2]) / (2 * lat_dis)
        # 前向差分（南边界）
        ddy[:, 0] = (dy[:, 1] - dy[:, 0]) / lat_dis
        # 后向差分（北边界）
        ddy[:, -1] = (dy[:, -1] - dy[:, -2]) / lat_dis
        return ddy

    #TODO
    def dd_z(self, dz,level):
        return 0


    def get_true_pressure(self,level):
        ds = self.ds.sel(level=level)
        T = ds["temperature"].values
        phi = ds["geopotential"].values
        p0 = ds["mean_sea_level_pressure"].values
        z = phi / self.g
        p_true = p0 * np.exp(-self.g * z / (R * T))
        return p_true