"""
使用重构模块的气象模拟代码
实现u风分量的数值预测和验证
"""

import xarray as xr
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from dataclasses import dataclass
import metpy.calc as mpcalc

# 假设已经导入了重构的模块
from Solver_weather.utils.weighted_acc_rmse import weighted_acc_torch_channels, weighted_rmse_torch
from pic_util import *
from WeatherDataProcessor import *

@dataclass
class SimulationConfig:
    """模拟配置参数"""
    file_path: str = "era5_day_2021-01-01.nc"
    level: int = 100
    diffusion_coefficient_flat: float = 10e3
    diffusion_coefficient_vertical: float = 1
    total_time: int = 24
    time_step: int = 1
    lat_range: Tuple[float, float] = (-50, 50)


class AdvectionCalculator:
    """平流项计算器"""

    def __init__(self, processor: 'WeatherDataProcessor'):
        self.processor = processor

    def calculate_u_advection_metpy(self, ds_metpy: xr.Dataset, level: int) -> np.ndarray:
        """使用MetPy计算u风分量的平流"""
        u = ds_metpy.sel(level=level)["u_component_of_wind"]
        v = ds_metpy.sel(level=level)["v_component_of_wind"]

        u_advection_metpy = mpcalc.advection(u, u=u, v=v, x_dim=-2, y_dim=-1, vertical_dim=-3)
        u_advection = u_advection_metpy.metpy.magnitude.copy()

        # 边界处理
        self._apply_boundary_correction(u_advection)

        return u_advection

    def _apply_boundary_correction(self, u_advection: np.ndarray):
        """对边界进行修正"""
        # 上下边界
        u_advection[0, :] /= 10000
        u_advection[-1, :] /= 10000

        # 左右边界（去掉已经除过的角点）
        u_advection[1:-1, 0] /= 10000
        u_advection[1:-1, -1] /= 10000

    def calculate_u_advection_manual(self, level: int) -> np.ndarray:
        """手动计算u风分量的平流"""
        # 使用重构的模块计算平流项
        duu_dx = self.processor.derivative.d_x(level,
                                               self.processor.wind.get_wind_by_type(level, WindType.UU))
        duv_dy = self.processor.derivative.d_y(level,
                                               self.processor.wind.get_wind_by_type(level, WindType.UV))

        # 垂直平流项
        u_lower = self.processor.wind.get_u_component(level)
        w_lower = self.processor.wind.get_w_component(level)
        u_upper = self.processor.wind.get_u_component(level + 50)
        w_upper = self.processor.wind.get_w_component(level + 50)

        duw_dz = self.processor.derivative.d_z([level, level + 50],
                                               u_lower * w_lower,
                                               u_upper * w_upper)

        return -(duu_dx + duv_dy + duw_dz)


class PressureGradientCalculator:
    """压力梯度力计算器"""

    def __init__(self, processor: 'WeatherDataProcessor'):
        self.processor = processor

    def calculate_pgf_geopotential(self, level: int) -> np.ndarray:
        """使用位势高度计算压力梯度力"""
        return self.processor.geopotential.get_gradient_x(level)

    def calculate_pgf_pressure(self, level: int) -> np.ndarray:
        """使用压力计算压力梯度力"""
        ds = self.processor.ds.sel(level=level)
        pressure = ds["pressure"].values if "pressure" in ds.data_vars else level * 100

        dp_dx = self.processor.derivative.d_x(level, pressure)
        rho = self.processor.density.get_density(level)

        return -(1 / rho) * dp_dx


class DiffusionCalculator:
    """扩散项计算器"""

    def __init__(self, processor: 'WeatherDataProcessor'):
        self.processor = processor

    def calculate_diffusion(self, level: int,
                            coeff_flat: float,
                            coeff_vertical: float) -> np.ndarray:
        """计算扩散项"""
        # 水平扩散
        u_wind = self.processor.wind.get_u_component(level)

        du_dx = self.processor.derivative.d_x(level, u_wind)
        du_dy = self.processor.derivative.d_y(level, u_wind)

        du_ddx = self.processor.derivative.d_x(level, du_dx)
        du_ddy = self.processor.derivative.d_y(level, du_dy)

        horizontal_diffusion = coeff_flat * (du_ddx + du_ddy)

        # 垂直扩散
        vertical_diffusion = self._calculate_vertical_diffusion(level, coeff_vertical)

        return horizontal_diffusion + vertical_diffusion

    def _calculate_vertical_diffusion(self, level: int, coeff: float) -> np.ndarray:
        """计算垂直扩散"""
        u_lower = self.processor.wind.get_u_component(level)
        u_mid = self.processor.wind.get_u_component(level + 50)
        u_upper = self.processor.wind.get_u_component(level + 100)

        # 计算一阶导数
        du_dz1 = self.processor.derivative.d_z([level, level + 50], u_lower, u_mid)
        du_dz2 = self.processor.derivative.d_z([level + 50, level + 100], u_mid, u_upper)

        # 计算二阶导数
        height_diff = self.processor.geopotential.get_height_difference(level, level + 50)
        du_ddz = (du_dz1 - du_dz2) / height_diff

        return coeff * du_ddz


class WeatherSimulator:
    """气象模拟器主类"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.ds = xr.open_dataset(config.file_path)
        self.processor = WeatherDataProcessor(self.ds)

        # 初始化各个计算器
        self.advection_calc = AdvectionCalculator(self.processor)
        self.pgf_calc = PressureGradientCalculator(self.processor)
        self.diffusion_calc = DiffusionCalculator(self.processor)

        # 存储计算结果
        self.results = {
            'du_dt': [],
            'u_current': [],
            'u_next': [],
            'time_indices': []
        }

    def run_simulation(self) -> Dict[str, List[np.ndarray]]:
        """运行模拟"""
        print("开始气象模拟...")

        for time_index in range(0, self.config.total_time, self.config.time_step):
            print(f"处理时间步 {time_index}/{self.config.total_time}")

            # 获取当前和下一时刻的数据
            time_curr = self.ds.time.values[time_index]
            time_next = self.ds.time.values[time_index] if time_index == 0 else self.ds.time.values[time_index]

            # 更新处理器的数据
            ds_curr = self.ds.sel(time=time_curr)
            ds_next = self.ds.sel(time=time_next)

            self.processor.ds = ds_curr
            self.processor.metpy_ds = ds_curr.metpy.parse_cf()

            # 计算各个物理过程
            du_dt = self._calculate_tendency(time_curr)
            u_current = self.processor.wind.get_u_component(self.config.level)

            # 获取下一时刻的观测值
            processor_next = WeatherDataProcessor(ds_next)
            u_next = processor_next.wind.get_u_component(self.config.level)

            # 存储结果
            self.results['du_dt'].append(du_dt)
            self.results['u_current'].append(u_current)
            self.results['u_next'].append(u_next)
            self.results['time_indices'].append(time_index)

        print("模拟完成！")
        return self.results

    def _calculate_tendency(self, time_curr) -> np.ndarray:
        """计算u风分量的时间变化率"""
        level = self.config.level

        # 1. 平流项
        ds_metpy = self.ds.metpy.parse_cf().sel(time=time_curr)
        u_advection = self.advection_calc.calculate_u_advection_metpy(ds_metpy, level)

        # 2. 压力梯度力
        pgf = self.pgf_calc.calculate_pgf_geopotential(level)

        # 3. 科里奥利力
        coriolis = self.processor.coriolis.get_u_coriolis_force(level)

        # 4. 扩散项
        diffusion = self.diffusion_calc.calculate_diffusion(
            level,
            self.config.diffusion_coefficient_flat,
            self.config.diffusion_coefficient_vertical
        )

        # 总的时间变化率
        du_dt = u_advection + pgf + coriolis + diffusion

        return du_dt

    def integrate_forward(self) -> np.ndarray:
        """前向积分预测"""
        print("开始前向积分...")

        du_dt_array = np.stack(self.results['du_dt'])
        u_next = np.stack(self.results['u_next'])

        # 初始化预测数组
        u_reconstructed = np.zeros_like(du_dt_array)
        u_reconstructed[0] = self.results['u_current'][0]

        # 简单的前向欧拉积分
        for t in range(1, len(du_dt_array)):
            dt = 3600  # 时间步长（秒）
            u_reconstructed[t] = u_reconstructed[t - 1] + du_dt_array[t - 1] * dt

        print("前向积分完成！")
        lat = self.processor.ds["latitude"]
        lon = self.processor.ds["longitude"]
        draw(u_reconstructed[-1],lat=lat,lon=lon,scale=1)
        draw(u_next[-1],lat=lat,lon=lon,scale=1)
        return u_reconstructed


class ValidationCalculator:
    """验证计算器"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.ds = xr.open_dataset(config.file_path)

    def calculate_metrics(self, u_pred: np.ndarray, u_true: np.ndarray) -> Tuple[List[float], List[float]]:
        """计算ACC和RMSE指标"""
        print("计算验证指标...")

        # 获取赤道区域的索引
        lat_vals = self.ds["latitude"].values
        lat_mask = (lat_vals >= self.config.lat_range[0]) & (lat_vals <= self.config.lat_range[1])
        lat_indices = np.where(lat_mask)[0]

        acc_list = []
        rmse_list = []

        for t in range(len(u_pred)):
            # 转换为torch张量
            pred = torch.from_numpy(u_pred[t][None, None]).float()
            true = torch.from_numpy(u_true[t][None, None]).float()

            # 选择赤道区域
            pred_eq = pred[:, :, :, lat_indices]
            true_eq = true[:, :, :, lat_indices]

            # 计算指标
            acc_t = weighted_acc_torch_channels(pred_eq, true_eq).item()
            rmse_t = weighted_rmse_torch(pred_eq, true_eq).item()

            acc_list.append(acc_t)
            rmse_list.append(rmse_t)

        print("验证指标计算完成！")
        return acc_list, rmse_list

    def plot_results(self, acc_list: List[float], rmse_list: List[float]):
        """绘制结果图表"""
        time_hours = np.arange(0, len(acc_list))

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(time_hours, acc_list, marker='o', linestyle='-', color='teal', linewidth=2)
        plt.title("ACC per Hour", fontsize=14, fontweight='bold')
        plt.xlabel("Hour", fontsize=12)
        plt.ylabel("ACC", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)

        plt.subplot(1, 2, 2)
        plt.plot(time_hours, rmse_list, marker='o', linestyle='-', color='darkorange', linewidth=2)
        plt.title("RMSE per Hour", fontsize=14, fontweight='bold')
        plt.xlabel("Hour", fontsize=12)
        plt.ylabel("RMSE", fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()


def main():
    """主函数"""
    # 配置参数
    config = SimulationConfig(
        file_path="era5_day_2021-01-01.nc",
        level=100,
        diffusion_coefficient_flat=10e3,
        diffusion_coefficient_vertical=1,
        total_time=24,
        time_step=1,
        lat_range=(-50, 50)
    )

    # 创建模拟器
    simulator = WeatherSimulator(config)

    # 运行模拟
    results = simulator.run_simulation()

    # 前向积分预测
    u_predicted = simulator.integrate_forward()

    # 获取真实值
    u_true = np.stack(results['u_next'][:len(u_predicted)])

    # 验证计算
    validator = ValidationCalculator(config)
    acc_list, rmse_list = validator.calculate_metrics(u_predicted, u_true)

    # 绘制结果
    validator.plot_results(acc_list, rmse_list)

    # 打印结果
    print("\n=== 模拟结果 ===")
    print(f"ACC列表: {acc_list}")
    print(f"RMSE列表: {rmse_list}")
    print(f"平均ACC: {np.mean(acc_list):.4f}")
    print(f"平均RMSE: {np.mean(rmse_list):.4f}")


if __name__ == "__main__":
    main()