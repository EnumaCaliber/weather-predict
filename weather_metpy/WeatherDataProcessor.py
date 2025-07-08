"""
重构后的气象数据处理模块
具有更好的扩展性和可维护性
"""

import xarray as xr
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional, Tuple
import metpy.calc as mpcalc
from metpy.units import units
from dataclasses import dataclass
from enum import Enum


@dataclass
class PhysicalConstants:
    """物理常数配置"""
    R: float = 287.0  # 气体常数
    P: float = 85000  # 参考压力
    TOP_MODEL: float = 10  # 模型顶层
    RADIUS: float = 6.371e6  # 地球半径
    OMEGA: float = 7.2921e-5  # 地球角速度
    G: float = 9.80665  # 重力加速度
    EARTH_RADIUS_M: float = 6370000  # 地球半径（米）


class WindType(Enum):
    """风场类型枚举"""
    U = "u"
    V = "v"
    W = "w"
    P = "p"
    Q = "q"
    UU = "uu"
    VU = "vu"
    WU = "wu"
    UV = "uv"
    VV = "vv"
    WV = "wv"
    UW = "uw"
    VW = "vw"
    WW = "ww"
    U_RHO = "urho"
    V_RHO = "vrho"
    W_RHO = "wrho"


class BaseDataProcessor(ABC):
    """数据处理基类"""

    def __init__(self, data_source: Union[str, xr.Dataset], constants: PhysicalConstants = None):
        self.constants = constants or PhysicalConstants()
        self._load_data(data_source)
        self._initialize_coordinates()

    def _load_data(self, data_source: Union[str, xr.Dataset]):
        """加载数据"""
        if isinstance(data_source, str):
            self.ds = xr.open_dataset(data_source)
        else:
            self.ds = data_source

        self.metpy_ds = self.ds.metpy.parse_cf()

    def _initialize_coordinates(self):
        """初始化坐标系参数"""
        self.cos_alpha = 1
        self.sin_alpha = 0

    @abstractmethod
    def get_variable(self, level: float, var_name: str) -> np.ndarray:
        """获取变量的抽象方法"""
        pass


class GeopotentialProcessor:
    """地势高度处理器"""

    def __init__(self, data_processor: BaseDataProcessor):
        self.processor = data_processor
        self.g = data_processor.constants.G

    def get_height(self, level: float) -> np.ndarray:
        """获取位势高度"""
        ds = self.processor.ds.sel(level=level)
        geopotential = ds["geopotential"].values
        return geopotential / self.g

    def get_height_difference(self, level1: float, level2: float) -> np.ndarray:
        """计算两个层次间的高度差"""
        ds1 = self.processor.ds.sel(level=level1)
        ds2 = self.processor.ds.sel(level=level2)

        geopotential1 = ds1["geopotential"].values
        geopotential2 = ds2["geopotential"].values

        return (geopotential1 - geopotential2) / self.g

    def get_gradient_x(self, level: float) -> np.ndarray:
        """计算经度方向的地势梯度"""
        height = self.get_height(level)
        distance_calc = DistanceCalculator(self.processor)
        lon_dis = distance_calc.get_longitude_distance(level)

        dz_dx = np.zeros_like(height)
        dz_dx[1:-1, :] = (height[2:, :] - height[:-2, :]) / (2 * lon_dis)
        dz_dx[0, :] = (height[1, :] - height[0, :]) / lon_dis
        dz_dx[-1, :] = (height[-1, :] - height[-2, :]) / lon_dis

        return dz_dx

    def get_gradient_y(self, level: float) -> np.ndarray:
        """计算纬度方向的地势梯度"""
        height = self.get_height(level)
        distance_calc = DistanceCalculator(self.processor)
        lat_dis = distance_calc.get_latitude_distance(level)

        dz_dy = np.zeros_like(height)
        dz_dy[:, 1:-1] = (height[:, 2:] - height[:, :-2]) / (2 * lat_dis)
        dz_dy[:, 0] = (height[:, 1] - height[:, 0]) / lat_dis
        dz_dy[:, -1] = (height[:, -1] - height[:, -2]) / lat_dis

        return -self.g * dz_dy  # 返回压力梯度力


class DistanceCalculator:
    """距离计算器"""

    def __init__(self, data_processor: BaseDataProcessor):
        self.processor = data_processor
        self.earth_radius = data_processor.constants.EARTH_RADIUS_M

    def get_latitude_distance(self, level: float) -> float:
        """计算纬度方向的距离"""
        ds = self.processor.ds.sel(level=level)
        lat = ds["latitude"].values
        delta_lat = lat[1] - lat[0]
        return delta_lat * 2 * np.pi * self.earth_radius / 360

    def get_longitude_distance(self, level: float) -> float:
        """计算经度方向的距离"""
        ds = self.processor.ds.sel(level=level)
        lon = ds["longitude"].values
        delta_lon = lon[1] - lon[0]
        return abs(delta_lon * 2 * np.pi * self.earth_radius / 360)


class WindProcessor:
    """风场处理器"""

    def __init__(self, data_processor: BaseDataProcessor):
        self.processor = data_processor
        self.g = data_processor.constants.G

    def get_u_component(self, level: float) -> np.ndarray:
        """获取u分量"""
        ds = self.processor.ds.sel(level=level)
        return ds["u_component_of_wind"].values

    def get_v_component(self, level: float) -> np.ndarray:
        """获取v分量"""
        ds = self.processor.ds.sel(level=level)
        return ds["v_component_of_wind"].values

    def get_w_component(self, level: float) -> np.ndarray:
        """获取w分量（垂直速度）"""
        ds = self.processor.ds.sel(level=level)
        w = ds["vertical_velocity"].values

        # 需要密度计算器来转换
        density_calc = DensityCalculator(self.processor)
        rho = density_calc.get_density(level)

        return -w / (rho * self.g)

    def get_wind_by_type(self, level: float, wind_type: WindType) -> np.ndarray:
        """根据类型获取风场数据"""
        wind_map = {
            WindType.U: lambda: self.get_u_component(level),
            WindType.V: lambda: self.get_v_component(level),
            WindType.W: lambda: self.get_w_component(level),
            WindType.UU: lambda: self.get_u_component(level) ** 2,
            WindType.VV: lambda: self.get_v_component(level) ** 2,
            WindType.WW: lambda: self.get_w_component(level) ** 2,
            WindType.UV: lambda: self.get_u_component(level) * self.get_v_component(level),
            WindType.UW: lambda: self.get_u_component(level) * self.get_w_component(level),
            WindType.VW: lambda: self.get_v_component(level) * self.get_w_component(level),
        }

        return wind_map[wind_type]()


class DensityCalculator:
    """密度计算器"""

    def __init__(self, data_processor: BaseDataProcessor):
        self.processor = data_processor

    def get_density(self, level: float) -> np.ndarray:
        """使用MetPy计算密度"""
        ds = self.processor.ds.sel(level=level)
        ds_metpy = self.processor.metpy_ds.sel(level=level)

        q = ds_metpy["specific_humidity"]
        T = ds_metpy["temperature"].metpy.convert_units('degC')

        rho = mpcalc.density(level * units.hPa, T, q * units('g/kg'))
        return rho.metpy.magnitude

    def get_virtual_temperature(self, level: float) -> np.ndarray:
        """计算虚温"""
        ds = self.processor.ds.sel(level=level)
        q = ds["specific_humidity"].values
        T = ds["temperature"].values
        return T * (1 + 0.61 * q)


class CoriolisProcessor:
    """科里奥利力处理器"""

    def __init__(self, data_processor: BaseDataProcessor):
        self.processor = data_processor
        self.omega = data_processor.constants.OMEGA

    def get_coriolis_parameter(self, level: float) -> np.ndarray:
        """获取科里奥利参数"""
        ds_metpy = self.processor.metpy_ds.sel(level=level)
        lat = ds_metpy['latitude'].values * units.degrees
        coriolis = mpcalc.coriolis_parameter(lat).magnitude

        # 扩展到经度维度
        lon_size = self.processor.ds.sel(level=level)["longitude"].values.size
        return np.tile(coriolis[np.newaxis, :], (lon_size, 1))

    def get_u_coriolis_force(self, level: float) -> np.ndarray:
        """计算u方向的科里奥利力"""
        wind_proc = WindProcessor(self.processor)
        v = wind_proc.get_v_component(level)
        coriolis = self.get_coriolis_parameter(level)
        return coriolis * v

    def get_v_coriolis_force(self, level: float) -> np.ndarray:
        """计算v方向的科里奥利力"""
        wind_proc = WindProcessor(self.processor)
        u = wind_proc.get_u_component(level)
        coriolis = self.get_coriolis_parameter(level)
        return -coriolis * u


class DerivativeCalculator:
    """导数计算器"""

    def __init__(self, data_processor: BaseDataProcessor):
        self.processor = data_processor
        self.distance_calc = DistanceCalculator(data_processor)

    def d_x(self, level: float, data: np.ndarray, order: int = 4) -> np.ndarray:
        """计算经度方向的导数"""
        lon_dis = self.distance_calc.get_longitude_distance(level)
        dx = np.zeros_like(data)

        if order == 4:
            dx[2:-2, :] = (-data[4:, :] + 8 * data[3:-1, :] - 8 * data[1:-3, :] + data[0:-4, :]) / (12 * lon_dis)
            # 边界处理
            dx[1, :] = (data[2, :] - data[0, :]) / (2 * lon_dis)
            dx[0, :] = (data[1, :] - data[0, :]) / lon_dis
            dx[-2, :] = (data[-1, :] - data[-3, :]) / (2 * lon_dis)
            dx[-1, :] = (data[-1, :] - data[-2, :]) / lon_dis
        elif order == 2:
            dx = (np.roll(data, -1, axis=0) - np.roll(data, 1, axis=0)) / (2 * lon_dis)

        return dx

    def d_y(self, level: float, data: np.ndarray, order: int = 4) -> np.ndarray:
        """计算纬度方向的导数"""
        lat_dis = self.distance_calc.get_latitude_distance(level)
        dy = np.zeros_like(data)

        if order == 4:
            dy[:, 2:-2] = (-data[:, 4:] + 8 * data[:, 3:-1] - 8 * data[:, 1:-3] + data[:, 0:-4]) / (12 * lat_dis)
            # 边界处理
            dy[:, 1] = (data[:, 2] - data[:, 0]) / (2 * lat_dis)
            dy[:, 0] = (data[:, 1] - data[:, 0]) / lat_dis
            dy[:, -2] = (data[:, -1] - data[:, -3]) / (2 * lat_dis)
            dy[:, -1] = (data[:, -1] - data[:, -2]) / lat_dis
        elif order == 2:
            dy[:, 1:-1] = (data[:, 2:] - data[:, :-2]) / (2 * lat_dis)
            dy[:, 0] = (data[:, 1] - data[:, 0]) / lat_dis
            dy[:, -1] = (data[:, -1] - data[:, -2]) / lat_dis

        return dy

    def d_z(self, levels: List[float], data1: np.ndarray, data2: np.ndarray) -> np.ndarray:
        """计算垂直方向的导数"""
        levels = sorted(levels)
        geopotential_proc = GeopotentialProcessor(self.processor)
        height_diff = geopotential_proc.get_height_difference(levels[0], levels[1])
        return (data2 - data1) / height_diff


class WeatherDataProcessor(BaseDataProcessor):
    """主要的气象数据处理器"""

    def __init__(self, data_source: Union[str, xr.Dataset], constants: PhysicalConstants = None):
        super().__init__(data_source, constants)

        # 初始化各个处理器
        self.geopotential = GeopotentialProcessor(self)
        self.wind = WindProcessor(self)
        self.density = DensityCalculator(self)
        self.coriolis = CoriolisProcessor(self)
        self.derivative = DerivativeCalculator(self)
        self.distance = DistanceCalculator(self)

    def get_variable(self, level: float, var_name: str) -> np.ndarray:
        """获取变量数据"""
        ds = self.ds.sel(level=level)
        return ds[var_name].values

    def get_coordinates(self, level: float) -> Dict[str, np.ndarray]:
        """获取坐标信息"""
        ds = self.ds.sel(level=level)
        return {
            'longitude': ds["longitude"].values,
            'latitude': ds["latitude"].values,
            'level': ds["level"].values
        }

    def get_pressure_gradient_force(self, level: float) -> Tuple[np.ndarray, np.ndarray]:
        """计算压力梯度力"""
        # 可以选择不同的方法计算压力梯度力
        pgf_x = self.geopotential.get_gradient_x(level)
        pgf_y = self.geopotential.get_gradient_y(level)
        return pgf_x, pgf_y

    def get_advection_term(self, level: float, variable: str) -> np.ndarray:
        """计算平流项"""
        u = self.wind.get_u_component(level)
        v = self.wind.get_v_component(level)
        var_data = self.get_variable(level, variable)

        d_var_dx = self.derivative.d_x(level, var_data)
        d_var_dy = self.derivative.d_y(level, var_data)

        return -(u * d_var_dx + v * d_var_dy)

    def calculate_vorticity(self, level: float) -> np.ndarray:
        """计算涡度"""
        u = self.wind.get_u_component(level)
        v = self.wind.get_v_component(level)

        du_dy = self.derivative.d_y(level, u)
        dv_dx = self.derivative.d_x(level, v)

        return dv_dx - du_dy

    def calculate_divergence(self, level: float) -> np.ndarray:
        """计算散度"""
        u = self.wind.get_u_component(level)
        v = self.wind.get_v_component(level)

        du_dx = self.derivative.d_x(level, u)
        dv_dy = self.derivative.d_y(level, v)

        return du_dx + dv_dy


# 使用示例
if __name__ == "__main__":
    # 创建处理器实例
    processor = WeatherDataProcessor("era5_day_2021-01-01.nc")

    # 获取850hPa的风场数据
    level = 850
    u_wind = processor.wind.get_u_component(level)
    v_wind = processor.wind.get_v_component(level)

    # 计算涡度和散度
    vorticity = processor.calculate_vorticity(level)
    divergence = processor.calculate_divergence(level)

    # 计算科里奥利力
    coriolis_u = processor.coriolis.get_u_coriolis_force(level)
    coriolis_v = processor.coriolis.get_v_coriolis_force(level)

    # 计算压力梯度力
    pgf_x, pgf_y = processor.get_pressure_gradient_force(level)

    print(f"数据处理完成，涡度范围: {vorticity.min():.2e} 到 {vorticity.max():.2e}")