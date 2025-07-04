import xarray as xr
from pic_util import *

# constant define
R = 287.0
P = 85000
top_model = 10
radius = 6.371e6


class get_point_parameters:
    def __init__(self, file_path):
        if isinstance(file_path, str):
            self.file_path = file_path
            self.ds = xr.open_dataset(file_path)
        else:
            self.ds = file_path
        self.cos_alpha = 1
        self.sin_alpha = 0
        self.omega = 7.2921e-5
        self.g = 9.80665

    def get_rho(self, level):
        ds = self.ds.sel(level=level)
        q = ds["specific_humidity"].values  # specific_humidity
        T = ds["temperature"].values
        P = self.get_true_pressure(level=level)
        rho = P / (R * T * (1 + 0.61 * q))
        return rho

    def get_t_virtual(self, level):
        ds = self.ds.sel(level=level)
        q = ds["specific_humidity"].values  # specific_humidity
        T = ds["temperature"].values
        T_virtual = T * (1 + 0.61 * q)
        return T_virtual

    def get_u_coriolis_force(self, level):
        ds = self.ds.sel(level=level)
        lon_size = ds["longitude"].values.size
        lat = ds["latitude"].values
        v = ds["v_component_of_wind"].values

        ##########################
        w = self.get_wind_w(level=level)
        ##########################

        lat = np.tile(lat[np.newaxis, :], (lon_size, 1))
        lat_rad = np.deg2rad(lat)
        fv = 2 * self.omega * np.sin(lat_rad) * v
        ew = 2 * self.omega * np.cos(lat_rad) * w * self.cos_alpha
        fw = 2 * self.omega * np.sin(lat_rad) * w * self.sin_alpha
        return fv + ew + fw

    def get_v_coriolis_force(self, level):
        ds = self.ds.sel(level=level)
        lon = ds["longitude"].values.size
        lat = ds["latitude"].values
        lat = np.tile(lat[np.newaxis, :], (lon, 1))

        u = ds["u_component_of_wind"].values
        w = self.get_wind_w(level=level)
        lat_rad = np.deg2rad(lat)
        fu = 2 * self.omega * np.sin(lat_rad) * u
        ew = 2 * self.omega * np.cos(lat_rad) * w * self.sin_alpha
        fw = 2 * self.omega * np.sin(lat_rad) * w * self.cos_alpha
        # return (- fu + ew + fw)
        return -fu

    def get_w_coriolis_force(self, level):
        ds = self.ds.sel(level=level)
        u = ds["u_component_of_wind"].values
        v = ds["v_component_of_wind"].values
        lon = ds["longitude"].values.size
        lat = ds["latitude"].values
        lat = np.tile(lat[np.newaxis, :], (lon, 1))
        lat_rad = np.deg2rad(lat)
        coriolis_force = 2 * self.omega * np.cos(lat_rad) * (u * self.cos_alpha + v * self.sin_alpha)
        return - coriolis_force

    def get_high_meter(self, level):
        ds = self.ds.sel(level=level)
        geopotential = ds["geopotential"].values
        high = geopotential / self.g
        return high

    def get_high_diff(self, level1, level2):
        ds_level1 = self.ds.sel(level=level1)
        ds_level2 = self.ds.sel(level=level2)
        geopotential1 = ds_level1["geopotential"].values
        geopotential2 = ds_level2["geopotential"].values

        high_diff = (geopotential1 - geopotential2) / self.g
        return high_diff

    def get_lat_distance(self, level):
        EARTH_RADIUS_M = 6370000
        ds = self.ds.sel(level=level)
        # meter
        lat = ds["latitude"].values
        delta_lat = lat[1] - lat[0]
        distance = delta_lat * 2 * np.pi * EARTH_RADIUS_M / 360
        return distance

    def get_lon_distance(self, level):
        EARTH_RADIUS_M = 6370000
        ds = self.ds.sel(level=level)
        # meter
        lon = ds["longitude"].values
        lat = ds["latitude"].values
        delta_lon = lon[1] - lon[0]
        distance = delta_lon * 2 * np.pi * EARTH_RADIUS_M / 360
        return abs(distance)

    def get_lon(self, level):
        ds = self.ds.sel(level=level)
        lon = ds["longitude"].values
        return lon

    def get_lat(self, level):
        ds = self.ds.sel(level=level)
        lat = ds["latitude"].values
        return lat

    def get_wind_u(self, level):
        ds = self.ds.sel(level=level)
        u = ds["u_component_of_wind"].values
        return u

    def get_wind_v(self, level):
        ds = self.ds.sel(level=level)
        v = ds["v_component_of_wind"].values
        return v

    def get_wind_w(self, level):
        ds = self.ds.sel(level=level)

        ############
        w = ds["vertical_velocity"].values
        rho = self.get_rho(level)
        w = - w / (rho * self.g)
        ############
        return w

    # lon
    def d_x(self, level, wind_type, order = 4):

        lon_dis = self.get_lon_distance(level=level)
        if wind_type == "u":
            wind = self.get_wind_u(level=level)
        elif wind_type == "v":
            wind = self.get_wind_v(level=level)
        elif wind_type == "w":
            wind = self.get_wind_w(level=level)
        elif wind_type == "p":
            wind = self.get_true_pressure(level=level)
        elif wind_type == "q":
            wind = self.get_specific_humidity(level=level)
        elif wind_type == "uu":
            wind = self.get_wind_u(level=level) * self.get_wind_u(level=level)
        elif wind_type == "vu":
            wind = self.get_wind_v(level=level) * self.get_wind_u(level=level)
        elif wind_type == "wu":
            wind = self.get_wind_w(level=level) * self.get_wind_u(level=level)
        elif wind_type == "urho":
            wind = self.get_rho(level=level) * self.get_wind_u(level=level)

        dx = np.zeros_like(wind)

        #
        # high order
        if order == 4:
            dx[2:-2, :] = (- wind[4:, :] + 8 * wind[3:-1, :] - 8 * wind[1:-3, :] + wind[0:-4, :]) / (12 * lon_dis)
            # 边界点使用二阶中心差分补充
            dx[1, :] = (wind[2, :] - wind[0, :]) / (2 * lon_dis)
            dx[0, :] = (wind[1, :] - wind[0, :]) / lon_dis
            dx[-2, :] = (wind[-1, :] - wind[-3, :]) / (2 * lon_dis)
            dx[-1, :] = (wind[-1, :] - wind[-2, :]) / lon_dis
        elif order == 2:
            dx = (np.roll(wind, -1, axis=0) - np.roll(wind, 1, axis=0)) / (2 * lon_dis)
        return dx

    # lat
    def d_y(self, level, wind_type,order = 4):
        lat_dis = self.get_lat_distance(level=level)
        if wind_type == "u":
            wind = self.get_wind_u(level=level)
        elif wind_type == "v":
            wind = self.get_wind_v(level=level)
        elif wind_type == "w":
            wind = self.get_wind_w(level=level)
        elif wind_type == "p":
            wind = self.get_true_pressure(level=level)
        elif wind_type == "q":
            wind = self.get_specific_humidity(level=level)
        elif wind_type == "uv":
            wind = self.get_wind_u(level=level) * self.get_wind_v(level=level)
        elif wind_type == "vv":
            wind = self.get_wind_v(level=level) * self.get_wind_v(level=level)
        elif wind_type == "wv":
            wind = self.get_wind_w(level=level) * self.get_wind_v(level=level)
        elif wind_type == "vrho":
            wind = self.get_rho(level=level) * self.get_wind_v(level=level)
        dy = np.zeros_like(wind)
        if order == 4:
            dy[:, 2:-2] = (- wind[:, 4:] + 8 * wind[:, 3:-1] - 8 * wind[:, 1:-3] + wind[:, 0:-4]) / (12 * lat_dis)
            # 边界点使用二阶中心差分
            dy[:, 1] = (wind[:, 2] - wind[:, 0]) / (2 * lat_dis)
            dy[:, 0] = (wind[:, 1] - wind[:, 0]) / lat_dis
            dy[:, -2] = (wind[:, -1] - wind[:, -3]) / (2 * lat_dis)
            dy[:, -1] = (wind[:, -1] - wind[:, -2]) / lat_dis
        elif order == 2:
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
        high_diff = self.get_high_diff(level1=level1, level2=level2)

        if wind_type == "u":
            wind1 = self.get_wind_u(level=level1)
            wind2 = self.get_wind_u(level=level2)
        elif wind_type == "v":
            wind1 = self.get_wind_v(level=level1)
            wind2 = self.get_wind_v(level=level2)
        elif wind_type == "w":
            wind1 = self.get_wind_w(level=level1)
            wind2 = self.get_wind_w(level=level2)
        elif wind_type == "p":
            wind1 = self.get_true_pressure(level=level1)
            wind2 = self.get_true_pressure(level=level2)
        elif wind_type == "q":
            wind1 = self.get_specific_humidity(level=level1)
            wind2 = self.get_specific_humidity(level=level2)
        elif wind_type == "uw":
            w1 = self.get_wind_w(level=level1)
            wind1 = self.get_wind_u(level=level1) * w1
            w2 = self.get_wind_w(level=level2)
            wind2 = self.get_wind_u(level=level2) * w2
        elif wind_type == "vw":
            w1 = self.get_wind_w(level=level1)
            wind1 = self.get_wind_v(level=level1) * w1
            w2 = self.get_wind_w(level=level2)
            wind2 = self.get_wind_v(level=level2) * w2
        elif wind_type == "wrho":
            w1 = self.get_wind_w(level=level1)
            w2 = self.get_wind_w(level=level2)
            wind1 = self.get_rho(level=level1) * w1
            wind2 = self.get_rho(level=level2) * w2
        else:
            w1 = self.get_wind_w(level=level1)
            wind1 = w1 * w1
            w2 = self.get_wind_w(level=level2)
            wind2 = w2 * w2
        dz = (wind2 - wind1) / high_diff
        return dz

    def dd_x(self, dx, level,order = 4):
        lon_dis = self.get_lon_distance(level=level)
        ddx = np.zeros_like(dx)
        if order == 4:
            ddx[2:-2, :] = (- dx[4:, :] + 8 * dx[3:-1, :] - 8 * dx[1:-3, :] + dx[0:-4, :]) / (12 * lon_dis)
            # 边界点使用二阶中心差分补充
            ddx[1, :] = (dx[2, :] - dx[0, :]) / (2 * lon_dis)
            ddx[0, :] = (dx[1, :] - dx[0, :]) / lon_dis
            ddx[-2, :] = (dx[-1, :] - dx[-3, :]) / (2 * lon_dis)
            ddx[-1, :] = (dx[-1, :] - dx[-2, :]) / lon_dis
        elif order == 2:
            ddx = (np.roll(dx, -1, axis=0) - np.roll(dx, 1, axis=0)) / (2 * lon_dis)
        return ddx

    def dd_y(self, dy, level,order = 4):
        lat_dis = self.get_lat_distance(level=level)
        ddy = np.zeros_like(dy)
        if order == 4:
            ddy[:, 2:-2] = (- dy[:, 4:] + 8 * dy[:, 3:-1] - 8 * dy[:, 1:-3] + dy[:, 0:-4]) / (12 * lat_dis)
            # 边界点使用二阶中心差分
            ddy[:, 1] = (dy[:, 2] - dy[:, 0]) / (2 * lat_dis)
            ddy[:, 0] = (dy[:, 1] - dy[:, 0]) / lat_dis
            ddy[:, -2] = (dy[:, -1] - dy[:, -3]) / (2 * lat_dis)
            ddy[:, -1] = (dy[:, -1] - dy[:, -2]) / lat_dis
        elif order == 2:
            ddy[:, 1:-1] = (dy[:, 2:] - dy[:, :-2]) / (2 * lat_dis)
            # 前向差分（南边界）
            ddy[:, 0] = (dy[:, 1] - dy[:, 0]) / lat_dis
            # 后向差分（北边界）
            ddy[:, -1] = (dy[:, -1] - dy[:, -2]) / lat_dis
        return ddy

    # du1 low level du2 high level
    def dd_z(self, level, du1, du2):
        level.sort()
        level1 = level[0]
        level2 = level[1]
        high_diff = self.get_high_diff(level1=level1, level2=level2)
        ddz = (du1 - du2) / (high_diff)
        return ddz

    def dd_z_center(self, levels):
        low, mid, high = sorted(levels)
        u_low = self.get_wind_u(level=low)
        u_mid = self.get_wind_u(level=mid)
        u_high = self.get_wind_u(level=high)
        z_low = self.get_high_meter(low)
        z_mid = self.get_high_meter(mid)
        z_high = self.get_high_meter(high)
        dz1 = z_mid - z_low
        dz2 = z_high - z_mid
        dz = (dz1 + dz2) / 2
        return (u_high - 2 * u_mid + u_low) / (dz ** 2)

    def get_true_pressure(self, level):
        ds = self.ds.sel(level=level)
        T = ds["temperature"].values
        phi = ds["geopotential"].values
        p0 = ds["mean_sea_level_pressure"].values
        z = phi / self.g
        p_true = p0 * np.exp(-self.g * z / (R * T))
        return p_true

    def get_level(self, level):
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


    def get_buoyancy(self, level):
        ds = self.ds.sel(level=level)
        T = ds["temperature"].values
        T_ref = ds["2m_temperature"].values
        Q = ds["specific_humidity"].values
        Tv = T * (1 + 0.61 * Q)

        B = self.g * (Tv - T_ref) / T_ref
        return B

    def eta_from_level(self, level):
        ds = self.ds.sel(level=level)
        pressure_k = level
        ps = ds["surface_pressure"].values
        eta_k = (pressure_k * 100 - top_model) / (ps - top_model)
        eta_k = np.where(ps >= pressure_k * 100, eta_k, np.nan)
        return eta_k

    def get_dp_dz_eta(self, level):
        ds = self.ds.sel(level=level)
        ps = ds["surface_pressure"].values
        level_higher = level + 25
        level_lower = level - 25

        eta_upper = self.eta_from_level(level_higher)
        eta_lower = self.eta_from_level(level_lower)
        z_upper = self.ds.sel(level=level_higher)["geopotential"].values / self.g
        z_lower = self.ds.sel(level=level_lower)["geopotential"].values / self.g
        d_eta_dz = (eta_upper - eta_lower) / (z_upper - z_lower)
        dp_dz = (ps - top_model) * d_eta_dz
        # dp_dz = np.where(ps > level * 100, dp_dz, np.nan)
        dp_dz = (ps >= level * 100).astype(float) * dp_dz
        return dp_dz

    def get_radius_effect(self, level):

        ds = self.ds.sel(level=level)
        u = ds["u_component_of_wind"].values
        v = ds["v_component_of_wind"].values
        ps = ds["geopotential"].values / self.g
        R = radius + ps
        return (u ** 2 + v ** 2) / R

    def get_specific_humidity(self, level):
        ds = self.ds.sel(level=level)
        q = ds["specific_humidity"].values
        return q

    def get_temperature(self, level):
        ds = self.ds.sel(level=level)
        T = ds["temperature"].values
        return T



    ####those methods maybe no useful
    def compute_dp_dz(self, level):
        """
        Approximate ∂p/∂z using finite difference between two pressure levels.
        """
        ds = self.ds
        p1 = level[0]
        p2 = level[1]
        ds1 = ds.sel(level=p1)
        ds2 = ds.sel(level=p2)

        # 单位转换为 Pa（如果是 hPa）

        p1_Pa = ds1["level"].values
        p2_Pa = ds2["level"].values

        delta_p = abs((p2_Pa - p1_Pa) * 100)  # scalar: pressure difference

        z1 = ds1["geopotential"].values / 9.80665  # convert from geopotential to meters
        z2 = ds2["geopotential"].values / 9.80665
        delta_z = z2 - z1  # shape: (lon, lat)

        dp_dz = delta_p / delta_z  # shape: (lon, lat)

        return dp_dz


    def get_surface_pressure(self, level):
        ds = self.ds.sel(level=level)
        ps = ds["surface_pressure"].values
        return ps


    def get_geopotential_dy(self, level):
        """
        计算纬度方向上的地势高度梯度：∂z/∂y，再乘以 -g 得到近似 PGF 项
        """
        g = 9.80665  # 重力加速度
        ds_level = self.ds.sel(level=level)

        geopotential = ds_level["geopotential"].values  # 单位: m^2/s^2
        height = geopotential / g  # 转换为 m

        lat = ds_level["latitude"].values
        delta_lat = np.deg2rad(lat[1] - lat[0])  # 纬度间隔，转为弧度
        R = 6370000  # 地球半径，单位 m
        dy = delta_lat * R  # 纬向格距，单位 m

        dz_dy = np.zeros_like(height)

        # 中心差分
        dz_dy[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / (2 * dy)
        dz_dy[:, 0] = (height[:, 1] - height[:, 0]) / dy
        dz_dy[:, -1] = (height[:, -1] - height[:, -2]) / dy

        pgf_y = -g * dz_dy
        return pgf_y  # 单位 m/s^2

    def get_geopotential_dx(self, level):
        """
        计算 geopotential height 的经度方向梯度（单位：m/m），考虑纬度引起的格距变化
        """
        ds = self.ds.sel(level=level)
        geopotential = ds["geopotential"].values  # shape: (lon, lat)
        height = geopotential / self.g  # 转换为位势高度 z，单位 m

        lon = ds["longitude"].values
        lat = ds["latitude"].values
        lon_size = len(lon)
        lat_size = len(lat)

        # 构造每个格点的经向距离（单位：米），考虑纬度变化
        delta_lon_deg = lon[1] - lon[0]  # 经度间隔（°）
        lat_rad = np.deg2rad(lat)  # shape: (lat,)
        dx = delta_lon_deg * (np.pi / 180) * radius  # shape: (lat,)
        # shape: (lon, lat)

        # 中心差分计算 ∂z/∂x
        dz_dx = np.zeros_like(height)
        dz_dx[1:-1, :] = (height[2:, :] - height[:-2, :]) / (2 * dx)
        dz_dx[0, :] = (height[1, :] - height[0, :]) / dx
        dz_dx[-1, :] = (height[-1, :] - height[-2, :]) / dx

        return dz_dx  # 单位：m/m