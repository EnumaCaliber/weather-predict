import xarray as xr
import numpy as np
from pic_util import *

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
        P = self.get_true_pressure(level=level)
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
        lon_size = ds["longitude"].values.size
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        v = ds["v_component_of_wind"].values


        ##########################
        w = ds["vertical_velocity"].values
        rho = self.get_rho(level)
        w = - w / (rho * self.g)
        print("w")
        draw(w,lon = lon , lat = lat,scale=1,title=f"w")
        ##########################

        lat = np.tile(lat[np.newaxis, :], (lon_size, 1))
        lat_rad = np.deg2rad(lat)
        fv = 2 * self.omega * np.sin(lat_rad) * v
        ew = 2 * self.omega * np.cos(lat_rad) * w * self.cos_alpha
        fw = 2 * self.omega * np.sin(lat_rad) * w * self.sin_alpha
        draw(fv, lon=lon, lat=lat, scale=1, title="fv")
        draw(ew, lon=lon, lat=lat, scale=1, title="ew")
        return (fv + ew + fw)

    def get_u_coriolis_force_mesh(self, level):
        ds = self.ds.sel(level=level)

        # 读取变量
        v = ds["v_component_of_wind"].values  # shape: (lon, lat) 确保！
        w = ds["vertical_velocity"].values
        rho = self.get_rho(level)
        w = - w / (rho * self.g)

        # 经纬网格生成
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        lon2d, lat2d = np.meshgrid(lon, lat, indexing="ij")  # shape: (lon, lat)
        lat_rad = np.deg2rad(lat2d)

        # 计算三项
        fv = 2 * self.omega * np.sin(lat_rad) * v
        ew = 2 * self.omega * np.cos(lat_rad) * w * self.cos_alpha
        fw = 2 * self.omega * np.sin(lat_rad) * w * self.sin_alpha

        # 可视化
        draw(fv, lon=lon, lat=lat, scale=1, title="fv")
        draw(ew, lon=lon, lat=lat, scale=1, title="ew")

        return fv + ew + fw

    def get_v_coriolis_force(self,level):
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

    def get_lon_distance(self,level):
        EARTH_RADIUS_M = 6370000
        ds = self.ds.sel(level=level)
        # meter
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        delta_lon = lon[1] - lon[0]
        # distance = delta_lon * 2 * np.pi *  EARTH_RADIUS_M / 360

        distance = delta_lon * 2 * np.pi * EARTH_RADIUS_M / 360
        return abs(distance)

    def get_lon(self,level):
        ds = self.ds.sel(level=level)
        lon = ds["longitude"].values
        return lon

    def get_lat(self,level):
        ds = self.ds.sel(level=level)
        lat = ds["latitude"].values
        return lat




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


        ############
        w = ds["vertical_velocity"].values
        rho = self.get_rho(level)
        w = - w / (rho * self.g)
        ############
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
        elif wind_type == "w":
            ########################
            wind = self.get_wind_w(level=level)
            rho = self.get_rho(level)
            wind = - wind / (rho * self.g)
            ########################
        elif wind_type == "p":
            wind = self.get_true_pressure(level=level)
            print("pressure")
        else:
            wind = self.get_wind_u(level=level) * self.get_wind_u(level=level)
            print("uu")
        dx = np.zeros_like(wind)
        dx = (np.roll(wind, -1, axis=0) - np.roll(wind, 1, axis=0)) / (2 * lon_dis)
        print("lon_dis:f{}",lon_dis)
        return dx


    # lat
    def d_y(self, level, wind_type):
        lat_dis = self.get_lat_distance(level=level)
        if wind_type == "u":
            wind = self.get_wind_u(level=level)
        elif wind_type == "v":
            wind = self.get_wind_v(level=level)
        elif wind_type == "w":
            ####################
            wind = self.get_wind_w(level=level)
            rho = self.get_rho(level)
            wind = - wind / (rho * self.g)
            ####################
        elif wind_type == "uv":
            wind = self.get_wind_u(level=level) * self.get_wind_v(level=level)
            print("uv")
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
        elif wind_type == "w":
            #########################################
            rho1 = self.get_rho(level=level1)
            wind1 = self.get_wind_w(level=level1)
            wind1 = - wind1 / (rho1 * self.g)
            rho2 = self.get_rho(level=level2)
            wind2 = self.get_wind_w(level=level2)
            wind2 = - wind2 / (rho2 * self.g)
            #########################################
        elif wind_type == "uw":
            rho1 = self.get_rho(level=level1)
            w1 = self.get_wind_w(level=level1)
            w1 = - w1 / (rho1 * self.g)
            wind1 = self.get_wind_u(level=level1) * w1

            rho2 = self.get_rho(level=level2)
            w2 = self.get_wind_w(level=level2)
            w2 = - w2 / (rho2 * self.g)
            wind2 = self.get_wind_u(level=level2) * w2
            print("uw")
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

    #du1 low level du2 high level
    def dd_z(self, level, du1,du2):
        level.sort()
        level1 = level[0]
        level2 = level[1]
        high_diff = self.get_high_diff(level1=level1,level2=level2)
        ddz = (du1 - du2) / (high_diff)
        return ddz


    def get_true_pressure(self,level):
        ds = self.ds.sel(level=level)
        T = ds["temperature"].values
        phi = ds["geopotential"].values
        p0 = ds["mean_sea_level_pressure"].values
        z = phi / self.g
        p_true = p0 * np.exp(-self.g * z / (R * T))
        return p_true

    def get_level(self,level):
        ds = self.ds.sel(level=level)
        level = ds["level"].values
        return level

    def get_geopotential_dx(self, level):
        ds = self.ds.sel(level=level)
        # 获取地势高度 (geopotential height)，单位 m
        geopotential = ds["geopotential"].values  # shape (lon, lat)
        height = geopotential / self.g  # 转换为 m
        lon_dis = self.get_lon_distance(level=level)

        # 中心差分计算 dz/dx
        dz_dx = np.zeros_like(height)
        dz_dx[1:-1, :] = (height[2:, :] - height[:-2, :]) / (2 * lon_dis)
        dz_dx[0, :] = (height[1, :] - height[0, :]) / lon_dis
        dz_dx[-1, :] = (height[-1, :] - height[-2, :]) / lon_dis

        return dz_dx  # 单位：m/m
