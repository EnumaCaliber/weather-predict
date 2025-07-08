import numpy as np
import xarray as xr
from scipy.interpolate import griddata, interp1d
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class ERA5GridProcessor:
    """
    ERA5数据网格处理器，模仿WRF的GEOGRID和METGRID功能
    """

    def __init__(self, config):
        self.config = config
        self.static_data = {}
        self.met_data = {}

    def setup_projection(self):
        """设置地图投影参数（类似WRF的map_proj）"""
        proj_config = self.config['projection']

        if proj_config['type'] == 'lambert':
            # Lambert conformal conic projection
            self.projection = {
                'type': 'lambert',
                'ref_lat': proj_config['ref_lat'],
                'ref_lon': proj_config['ref_lon'],
                'truelat1': proj_config['truelat1'],
                'truelat2': proj_config['truelat2']
            }
        elif proj_config['type'] == 'mercator':
            self.projection = {
                'type': 'mercator',
                'ref_lat': proj_config['ref_lat'],
                'ref_lon': proj_config['ref_lon']
            }
        else:
            # 默认使用经纬度网格
            self.projection = {'type': 'latlon'}

    def create_target_grid(self):
        """创建目标网格（类似GEOGRID功能）"""
        domain = self.config['domain']

        # 检查是否为全球网格
        is_global = (domain['west_lon'] == -180.0 and domain['east_lon'] == 180.0 and
                     domain['south_lat'] == -90.0 and domain['north_lat'] == 90.0)

        if is_global:
            print("Creating global grid...")

        # 创建目标网格坐标
        if self.projection['type'] == 'latlon':
            # 经纬度网格
            if is_global:
                # 全球网格特殊处理
                lon_1d = np.linspace(domain['west_lon'], domain['east_lon'], domain['nx'], endpoint=False)
                lat_1d = np.linspace(domain['south_lat'], domain['north_lat'], domain['ny'])
            else:
                # 区域网格
                lon_1d = np.linspace(domain['west_lon'], domain['east_lon'], domain['nx'])
                lat_1d = np.linspace(domain['south_lat'], domain['north_lat'], domain['ny'])

            self.target_lon, self.target_lat = np.meshgrid(lon_1d, lat_1d)
        else:
            # 投影网格（简化处理）
            x_1d = np.linspace(0, domain['nx'] - 1, domain['nx']) * domain['dx']
            y_1d = np.linspace(0, domain['ny'] - 1, domain['ny']) * domain['dy']
            x_grid, y_grid = np.meshgrid(x_1d, y_1d)

            # 转换为经纬度（这里简化处理，实际应用需要完整的投影变换）
            self.target_lon = domain['ref_lon'] + x_grid / 111000.0  # 近似转换
            self.target_lat = domain['ref_lat'] + y_grid / 111000.0

        # 计算网格间距
        if is_global:
            # 全球网格的网格间距随纬度变化
            self.dx = 2 * np.pi * 6371000 / domain['nx']  # 赤道处的网格间距
            self.dy = np.pi * 6371000 / (domain['ny'] - 1)  # 纬度方向网格间距
        else:
            self.dx = domain.get('dx', 25000)  # 默认25km
            self.dy = domain.get('dy', 25000)

        # 创建垂直网格（类似WRF的eta坐标）
        self.nz = domain.get('nz', 50)
        self.eta_levels = self.create_eta_levels()

        # 全球网格信息
        if is_global:
            print(f"Global grid created: {domain['nx']} x {domain['ny']} x {self.nz}")
            print(f"Resolution: {360 / domain['nx']:.3f}° x {180 / (domain['ny'] - 1):.3f}°")
            print(f"Equatorial grid spacing: dx={self.dx / 1000:.1f}km, dy={self.dy / 1000:.1f}km")
        else:
            print(f"Regional grid created: {domain['nx']} x {domain['ny']} x {self.nz}")
            print(f"Grid spacing: dx={self.dx / 1000:.1f}km, dy={self.dy / 1000:.1f}km")

        # 存储全球网格标识
        self.is_global = is_global

    def create_eta_levels(self):
        """创建eta垂直坐标（地形跟随坐标）"""
        # 类似WRF的eta坐标分布
        eta_levels = np.zeros(self.nz + 1)  # 层界面

        # 底层密集，顶层稀疏的分布
        for k in range(self.nz + 1):
            eta = k / self.nz
            # 使用双曲正切函数创建非均匀分布
            eta_levels[k] = np.tanh(2.5 * eta) / np.tanh(2.5)

        return eta_levels

    def load_era5_data(self, era5_file):
        """加载ERA5数据"""
        print("Loading ERA5 data...")

        # 读取ERA5数据
        ds = xr.open_dataset(era5_file)

        # 打印数据集信息用于调试
        print(f"Dataset dimensions: {dict(ds.dims)}")
        print(f"Dataset variables: {list(ds.data_vars.keys())}")
        print(f"Dataset coordinates: {list(ds.coords.keys())}")

        # 标准化坐标名称（ERA5可能使用不同的名称）
        coord_mapping = {
            'longitude': ['longitude', 'lon', 'long'],
            'latitude': ['latitude', 'lat'],
            'level': ['level', 'plev', 'pressure_level'],
            'time': ['time', 'valid_time']
        }

        coords = {}
        for standard_name, possible_names in coord_mapping.items():
            for name in possible_names:
                if name in ds.coords:
                    coords[standard_name] = name
                    break

        # 提取坐标信息
        if 'longitude' in coords:
            self.era5_lon = ds[coords['longitude']].values
            print(f"Longitude range: {self.era5_lon.min():.2f} to {self.era5_lon.max():.2f}")

        if 'latitude' in coords:
            self.era5_lat = ds[coords['latitude']].values
            print(f"Latitude range: {self.era5_lat.min():.2f} to {self.era5_lat.max():.2f}")

        if 'level' in coords:
            self.era5_levels = ds[coords['level']].values
            print(f"Pressure levels: {self.era5_levels}")
        else:
            self.era5_levels = None
            print("No pressure levels found - surface data only")

        if 'time' in coords:
            self.era5_time = ds[coords['time']].values
            print(f"Time steps: {len(self.era5_time)}")

        # 识别可能的变量名称
        var_mapping = {
            't': ['t', 'temperature', 'T', 'air_temperature'],
            'u': ['u', 'u_component_of_wind', 'U', 'eastward_wind'],
            'v': ['v', 'v_component_of_wind', 'V', 'northward_wind'],
            'q': ['q', 'specific_humidity', 'Q', 'humidity'],
            'z': ['z', 'geopotential', 'Z', 'geopotential_height'],
            'sp': ['sp', 'surface_pressure', 'ps', 'pressure'],
            'msl': ['msl', 'mean_sea_level_pressure', 'slp']
        }

        # 提取所需变量
        self.era5_data = {}
        for standard_name, possible_names in var_mapping.items():
            for name in possible_names:
                if name in ds.data_vars:
                    self.era5_data[standard_name] = ds[name]
                    print(f"Found variable: {name} -> {standard_name}")
                    break

        print(f"ERA5 data loaded: {len(self.era5_lon)} x {len(self.era5_lat)} grid")
        print(f"Variables found: {list(self.era5_data.keys())}")

        return ds

    def interpolate_horizontal(self, era5_var, time_idx=0):
        """水平插值到目标网格（类似METGRID功能）"""
        print(f"Horizontal interpolation for variable: {era5_var.name}")

        # 处理不同维度的数据
        if len(era5_var.dims) == 4:  # (time, level, lat, lon)
            data = era5_var.isel(time=time_idx).values
            interpolated = np.zeros((len(self.era5_levels), self.config['domain']['ny'], self.config['domain']['nx']))

            # 创建ERA5的经纬度网格
            era5_lon_mesh, era5_lat_mesh = np.meshgrid(self.era5_lon, self.era5_lat)

            # 处理全球网格的周期性边界条件
            if hasattr(self, 'is_global') and self.is_global:
                # 扩展ERA5数据以处理180°/-180°边界
                era5_lon_extended = np.concatenate([self.era5_lon - 360, self.era5_lon, self.era5_lon + 360])
                era5_lon_mesh_ext, era5_lat_mesh_ext = np.meshgrid(era5_lon_extended, self.era5_lat)

                # 扩展数据
                for k, level in enumerate(self.era5_levels):
                    print(f"  Processing level {level} hPa ({k + 1}/{len(self.era5_levels)})")

                    # 扩展数据以处理周期性边界
                    data_extended = np.concatenate([data[k], data[k], data[k]], axis=1)

                    # 创建插值点
                    points = np.column_stack([era5_lon_mesh_ext.flatten(), era5_lat_mesh_ext.flatten()])
                    target_points = np.column_stack([self.target_lon.flatten(), self.target_lat.flatten()])

                    # 插值数据
                    data_flat = data_extended.flatten()
                    mask = ~np.isnan(data_flat)

                    if np.sum(mask) > 10:
                        try:
                            interpolated_flat = griddata(points[mask], data_flat[mask],
                                                         target_points, method='linear', fill_value=np.nan)
                            interpolated[k] = interpolated_flat.reshape(self.config['domain']['ny'],
                                                                        self.config['domain']['nx'])

                            # 填充NaN值
                            nan_mask = np.isnan(interpolated[k])
                            if np.any(nan_mask):
                                interpolated_nearest = griddata(points[mask], data_flat[mask],
                                                                target_points, method='nearest')
                                interpolated_nearest = interpolated_nearest.reshape(self.config['domain']['ny'],
                                                                                    self.config['domain']['nx'])
                                interpolated[k][nan_mask] = interpolated_nearest[nan_mask]

                        except Exception as e:
                            print(f"    Warning: Interpolation failed for level {level}: {e}")
                            interpolated[k] = np.full((self.config['domain']['ny'], self.config['domain']['nx']),
                                                      np.nan)
                    else:
                        print(f"    Warning: Not enough valid data points for level {level}")
                        interpolated[k] = np.full((self.config['domain']['ny'], self.config['domain']['nx']), np.nan)
            else:
                # 区域网格的常规处理
                for k, level in enumerate(self.era5_levels):
                    print(f"  Processing level {level} hPa ({k + 1}/{len(self.era5_levels)})")

                    points = np.column_stack([era5_lon_mesh.flatten(), era5_lat_mesh.flatten()])
                    target_points = np.column_stack([self.target_lon.flatten(), self.target_lat.flatten()])

                    data_flat = data[k].flatten()
                    mask = ~np.isnan(data_flat)

                    if np.sum(mask) > 10:
                        try:
                            interpolated_flat = griddata(points[mask], data_flat[mask],
                                                         target_points, method='linear', fill_value=np.nan)
                            interpolated[k] = interpolated_flat.reshape(self.config['domain']['ny'],
                                                                        self.config['domain']['nx'])

                            nan_mask = np.isnan(interpolated[k])
                            if np.any(nan_mask):
                                interpolated_nearest = griddata(points[mask], data_flat[mask],
                                                                target_points, method='nearest')
                                interpolated_nearest = interpolated_nearest.reshape(self.config['domain']['ny'],
                                                                                    self.config['domain']['nx'])
                                interpolated[k][nan_mask] = interpolated_nearest[nan_mask]

                        except Exception as e:
                            print(f"    Warning: Interpolation failed for level {level}: {e}")
                            interpolated[k] = np.full((self.config['domain']['ny'], self.config['domain']['nx']),
                                                      np.nan)
                    else:
                        print(f"    Warning: Not enough valid data points for level {level}")
                        interpolated[k] = np.full((self.config['domain']['ny'], self.config['domain']['nx']), np.nan)

        elif len(era5_var.dims) == 3:  # (time, lat, lon) - 地面变量
            data = era5_var.isel(time=time_idx).values

            # 创建ERA5的经纬度网格
            era5_lon_mesh, era5_lat_mesh = np.meshgrid(self.era5_lon, self.era5_lat)

            # 处理全球网格
            if hasattr(self, 'is_global') and self.is_global:
                # 扩展数据处理周期性边界
                era5_lon_extended = np.concatenate([self.era5_lon - 360, self.era5_lon, self.era5_lon + 360])
                era5_lon_mesh_ext, era5_lat_mesh_ext = np.meshgrid(era5_lon_extended, self.era5_lat)
                data_extended = np.concatenate([data, data, data], axis=1)

                points = np.column_stack([era5_lon_mesh_ext.flatten(), era5_lat_mesh_ext.flatten()])
                data_flat = data_extended.flatten()
            else:
                # 区域网格
                points = np.column_stack([era5_lon_mesh.flatten(), era5_lat_mesh.flatten()])
                data_flat = data.flatten()

            target_points = np.column_stack([self.target_lon.flatten(), self.target_lat.flatten()])
            mask = ~np.isnan(data_flat)

            if np.sum(mask) > 10:
                try:
                    interpolated_flat = griddata(points[mask], data_flat[mask],
                                                 target_points, method='linear', fill_value=np.nan)
                    interpolated = interpolated_flat.reshape(self.config['domain']['ny'], self.config['domain']['nx'])

                    nan_mask = np.isnan(interpolated)
                    if np.any(nan_mask):
                        interpolated_nearest = griddata(points[mask], data_flat[mask],
                                                        target_points, method='nearest')
                        interpolated_nearest = interpolated_nearest.reshape(self.config['domain']['ny'],
                                                                            self.config['domain']['nx'])
                        interpolated[nan_mask] = interpolated_nearest[nan_mask]

                except Exception as e:
                    print(f"    Warning: Interpolation failed: {e}")
                    interpolated = np.full((self.config['domain']['ny'], self.config['domain']['nx']), np.nan)
            else:
                print(f"    Warning: Not enough valid data points")
                interpolated = np.full((self.config['domain']['ny'], self.config['domain']['nx']), np.nan)

        else:
            print(f"    Warning: Unsupported data dimensions: {era5_var.dims}")
            interpolated = None

        return interpolated

    def interpolate_vertical(self, pressure_data, target_eta):
        """垂直插值到eta坐标（类似WRF的垂直坐标变换）"""
        print("Vertical interpolation to eta coordinates...")

        ny, nx = self.config['domain']['ny'], self.config['domain']['nx']
        nz_target = len(target_eta) - 1

        # 为每个变量创建eta坐标上的数据
        interpolated_data = {}

        # 获取地面压力
        surface_pressure = None
        if hasattr(self, 'surface_pressure') and self.surface_pressure is not None:
            surface_pressure = self.surface_pressure
        elif 'sp' in pressure_data:
            surface_pressure = pressure_data['sp']

        if surface_pressure is None:
            print("Warning: No surface pressure data found, using standard atmosphere")
            surface_pressure = np.full((ny, nx), 101325.0)  # 标准大气压

        # 确保地面压力是2D数组
        if len(surface_pressure.shape) == 3:
            surface_pressure = surface_pressure[0]  # 取第一个时间步

        for var_name, var_data in pressure_data.items():
            if len(var_data.shape) == 3:  # 3D变量 (level, lat, lon)
                var_eta = np.zeros((nz_target, ny, nx))

                print(f"  Processing {var_name} with shape {var_data.shape}")

                for j in range(ny):
                    for i in range(nx):
                        # 计算每层的压力
                        ps = surface_pressure[j, i]  # 地面压力
                        ptop = 5000.0  # 模式顶压力 (Pa)

                        # eta坐标对应的压力
                        p_eta = ptop + target_eta[:-1] * (ps - ptop)  # 层中心压力

                        # 插值到eta坐标
                        valid_mask = ~np.isnan(var_data[:, j, i])
                        if np.sum(valid_mask) > 1:
                            try:
                                # ERA5压力层是hPa，转换为Pa
                                pressure_levels_pa = self.era5_levels * 100

                                # 确保压力层是递减的（从高压到低压）
                                if pressure_levels_pa[0] < pressure_levels_pa[-1]:
                                    pressure_levels_pa = pressure_levels_pa[::-1]
                                    var_profile = var_data[::-1, j, i]
                                    valid_mask = valid_mask[::-1]
                                else:
                                    var_profile = var_data[:, j, i]

                                # 只使用有效数据点进行插值
                                valid_pressures = pressure_levels_pa[valid_mask]
                                valid_values = var_profile[valid_mask]

                                if len(valid_pressures) > 1:
                                    f = interp1d(valid_pressures, valid_values,
                                                 kind='linear', bounds_error=False,
                                                 fill_value='extrapolate')
                                    var_eta[:, j, i] = f(p_eta)

                            except Exception as e:
                                print(f"    Warning: Interpolation failed at ({j},{i}): {e}")
                                var_eta[:, j, i] = np.nan

                interpolated_data[var_name] = var_eta

            elif len(var_data.shape) == 2:  # 2D变量 (lat, lon)
                interpolated_data[var_name] = var_data

        return interpolated_data

    def apply_terrain_following(self, surface_elevation):
        """应用地形跟随坐标系"""
        print("Applying terrain-following coordinates...")

        # 验证输入维度
        expected_shape = (self.config['domain']['ny'], self.config['domain']['nx'])
        if surface_elevation.shape != expected_shape:
            print(f"Warning: Surface elevation shape {surface_elevation.shape} doesn't match expected {expected_shape}")
            if len(surface_elevation.shape) == 3:
                # 如果是3D数据，取最底层或平均
                surface_elevation = surface_elevation[-1]  # 取最底层
                print(f"Using bottom layer, new shape: {surface_elevation.shape}")
            elif len(surface_elevation.shape) == 2:
                # 如果维度正确但形状不对，进行reshape或插值
                if surface_elevation.size == expected_shape[0] * expected_shape[1]:
                    surface_elevation = surface_elevation.reshape(expected_shape)
                else:
                    print(f"Creating zero terrain due to dimension mismatch")
                    surface_elevation = np.zeros(expected_shape)
            else:
                print(f"Creating zero terrain due to unsupported dimensions")
                surface_elevation = np.zeros(expected_shape)

        # 平滑地形（防止数值不稳定）
        smoothed_terrain = gaussian_filter(surface_elevation, sigma=1.0)

        # 计算地形高度对网格的影响
        ny, nx = self.config['domain']['ny'], self.config['domain']['nx']

        # 创建3D网格高度
        grid_height = np.zeros((self.nz, ny, nx))

        for k in range(self.nz):
            eta = self.eta_levels[k]
            # 地形跟随：z = eta * (地面高度) + (1-eta) * (模式顶高度)
            model_top = 20000.0  # 20km模式顶
            grid_height[k] = eta * smoothed_terrain + (1 - eta) * model_top

        print(f"Grid height shape: {grid_height.shape}")
        print(f"Terrain height range: {smoothed_terrain.min():.1f} to {smoothed_terrain.max():.1f} m")

        return grid_height, smoothed_terrain

    def calculate_map_factors(self):
        """计算地图投影因子（类似WRF的map factors）"""
        print("Calculating map factors...")

        if self.projection['type'] == 'lambert':
            # Lambert投影的地图因子
            ref_lat = np.radians(self.projection['ref_lat'])
            truelat1 = np.radians(self.projection['truelat1'])
            truelat2 = np.radians(self.projection['truelat2'])

            # 简化的地图因子计算
            map_factor = np.cos(ref_lat) / np.cos(np.radians(self.target_lat))
        else:
            # 经纬度网格的地图因子
            map_factor = np.cos(np.radians(self.target_lat))

        return map_factor

    def process_era5_to_wrf_grid(self, era5_file, output_file=None):
        """完整的ERA5到WRF网格处理流程"""
        print("Starting ERA5 to WRF grid processing...")

        # 1. 设置投影和目标网格
        self.setup_projection()
        self.create_target_grid()

        # 2. 加载ERA5数据
        era5_ds = self.load_era5_data(era5_file)

        # 3. 水平插值
        print("\n=== Starting horizontal interpolation ===")
        interpolated_vars = {}
        for var_name, var_data in self.era5_data.items():
            print(f"Processing variable: {var_name}")
            interpolated_vars[var_name] = self.interpolate_horizontal(var_data)
            if interpolated_vars[var_name] is not None:
                print(f"  Result shape: {interpolated_vars[var_name].shape}")
            else:
                print(f"  Failed to interpolate {var_name}")

        # 4. 存储地面压力（用于垂直插值）
        print("\n=== Setting up surface pressure ===")
        if 'sp' in interpolated_vars and interpolated_vars['sp'] is not None:
            self.surface_pressure = interpolated_vars['sp']
            print(f"Surface pressure shape: {self.surface_pressure.shape}")
        else:
            print("Warning: No surface pressure data found")

        # 5. 垂直插值到eta坐标
        print("\n=== Starting vertical interpolation ===")
        eta_data = self.interpolate_vertical(interpolated_vars, self.eta_levels)

        # 6. 应用地形跟随坐标
        if 'z' in interpolated_vars:
            # 使用地面位势高度
            z_data = interpolated_vars['z']
            if len(z_data.shape) == 3:
                # 如果是3D数据，取最底层（最高压力层）
                surface_elevation = z_data[-1] / 9.81  # 转换为高度
            else:
                # 如果是2D数据，直接使用
                surface_elevation = z_data / 9.81
            grid_height, smoothed_terrain = self.apply_terrain_following(surface_elevation)
        else:
            # 使用平坦地形
            grid_height, smoothed_terrain = self.apply_terrain_following(
                np.zeros((self.config['domain']['ny'], self.config['domain']['nx']))
            )

        # 7. 计算地图因子
        map_factor = self.calculate_map_factors()

        # 8. 创建输出数据结构
        output_data = {
            'coordinates': {
                'longitude': self.target_lon,
                'latitude': self.target_lat,
                'eta_levels': self.eta_levels,
                'grid_height': grid_height
            },
            'static_fields': {
                'terrain_height': smoothed_terrain,
                'map_factor': map_factor,
                'dx': self.dx,
                'dy': self.dy
            },
            'meteorological_fields': eta_data,
            'projection': self.projection
        }

        # 9. 保存结果
        if output_file:
            self.save_output(output_data, output_file)

        print("Processing completed successfully!")
        return output_data

    def save_output(self, data, filename):
        """保存处理后的数据"""
        print(f"Saving output to {filename}...")

        # 创建xarray Dataset
        coords = {
            'eta': self.eta_levels[:-1],  # 层中心
            'eta_stag': self.eta_levels,  # 层界面
            'lat': np.arange(self.config['domain']['ny']),
            'lon': np.arange(self.config['domain']['nx'])
        }

        data_vars = {}

        # 添加坐标变量
        data_vars['longitude'] = (['lat', 'lon'], data['coordinates']['longitude'])
        data_vars['latitude'] = (['lat', 'lon'], data['coordinates']['latitude'])
        data_vars['terrain'] = (['lat', 'lon'], data['static_fields']['terrain_height'])
        data_vars['map_factor'] = (['lat', 'lon'], data['static_fields']['map_factor'])

        # 添加气象变量
        for var_name, var_data in data['meteorological_fields'].items():
            if len(var_data.shape) == 3:
                data_vars[var_name] = (['eta', 'lat', 'lon'], var_data)

        # 创建Dataset
        ds = xr.Dataset(data_vars, coords=coords)

        # 添加属性
        ds.attrs['projection'] = str(data['projection'])
        ds.attrs['dx'] = data['static_fields']['dx']
        ds.attrs['dy'] = data['static_fields']['dy']
        ds.attrs['created'] = datetime.now().isoformat()

        # 保存为NetCDF文件
        ds.to_netcdf(filename)
        print(f"Output saved to {filename}")

    def plot_results(self, data, var_name='t', level=0):
        """绘制结果"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # 地形
        im1 = axes[0].contourf(data['coordinates']['longitude'],
                               data['coordinates']['latitude'],
                               data['static_fields']['terrain_height'],
                               levels=20, cmap='terrain')
        axes[0].set_title('Terrain Height (m)')
        axes[0].set_xlabel('Longitude')
        axes[0].set_ylabel('Latitude')
        plt.colorbar(im1, ax=axes[0])

        # 气象变量
        if var_name in data['meteorological_fields']:
            var_data = data['meteorological_fields'][var_name]
            if len(var_data.shape) == 3:
                im2 = axes[1].contourf(data['coordinates']['longitude'],
                                       data['coordinates']['latitude'],
                                       var_data[level],
                                       levels=20, cmap='viridis')
                axes[1].set_title(f'{var_name} at level {level}')
                axes[1].set_xlabel('Longitude')
                axes[1].set_ylabel('Latitude')
                plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()


# 使用示例
def create_test_era5_data(filename):
    """创建测试用的ERA5格式数据"""
    print("Creating test ERA5 data...")

    # 创建测试网格
    lon = np.linspace(-180, 179.25, 240)  # 1.5度分辨率
    lat = np.linspace(90, -90, 121)  # 1.5度分辨率
    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10])
    time = [np.datetime64('2024-01-01T00:00:00')]

    # 创建测试数据
    np.random.seed(42)  # 确保可重复性

    # 创建经纬度网格
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    # 生成测试数据
    data_vars = {}

    # 3D变量 (time, level, lat, lon)
    for var in ['t', 'u', 'v', 'q', 'z']:
        data_3d = np.zeros((1, len(levels), len(lat), len(lon)))
        for k, level in enumerate(levels):
            if var == 't':  # 温度
                data_3d[0, k] = 273.15 + 15 * np.exp(-level / 1000) + 10 * np.sin(np.radians(lat_mesh)) * np.cos(
                    np.radians(lon_mesh))
            elif var == 'u':  # 东西风
                data_3d[0, k] = 20 * np.sin(np.radians(lat_mesh * 2)) * (1000 - level) / 1000
            elif var == 'v':  # 南北风
                data_3d[0, k] = 10 * np.cos(np.radians(lon_mesh)) * (1000 - level) / 1000
            elif var == 'q':  # 比湿
                data_3d[0, k] = 0.01 * np.exp(-level / 500) * np.exp(-np.abs(lat_mesh) / 30)
            elif var == 'z':  # 位势高度
                data_3d[0, k] = 9.81 * (16000 - level * 10) * (1 + 0.1 * np.sin(np.radians(lat_mesh)))

        data_vars[var] = (['time', 'level', 'latitude', 'longitude'], data_3d)

    # 2D变量 (time, lat, lon)
    data_vars['sp'] = (['time', 'latitude', 'longitude'],
                       np.array([101325 * (1 - 0.1 * np.sin(np.radians(lat_mesh)))]).reshape(1, len(lat), len(lon)))

    # 创建坐标
    coords = {
        'time': time,
        'level': levels,
        'latitude': lat,
        'longitude': lon
    }

    # 创建Dataset
    ds = xr.Dataset(data_vars, coords=coords)

    # 添加属性
    ds['t'].attrs = {'units': 'K', 'long_name': 'Temperature'}
    ds['u'].attrs = {'units': 'm/s', 'long_name': 'U component of wind'}
    ds['v'].attrs = {'units': 'm/s', 'long_name': 'V component of wind'}
    ds['q'].attrs = {'units': 'kg/kg', 'long_name': 'Specific humidity'}
    ds['z'].attrs = {'units': 'm^2/s^2', 'long_name': 'Geopotential'}
    ds['sp'].attrs = {'units': 'Pa', 'long_name': 'Surface pressure'}

    # 保存文件
    ds.to_netcdf(filename)
    print(f"Test ERA5 data saved to {filename}")
    return ds


def main():
    # 全球配置选项
    configs = {
        'global_coarse': {
            'domain': {
                'nx': 360,  # 1度分辨率
                'ny': 181,  # 1度分辨率
                'nz': 30,
                'dx': 111000,  # 约111km
                'dy': 111000,
                'west_lon': -180.0,
                'east_lon': 180.0,
                'south_lat': -90.0,
                'north_lat': 90.0,
                'ref_lon': 0.0,
                'ref_lat': 0.0
            },
            'projection': {'type': 'latlon'}
        },

        'global_medium': {
            'domain': {
                'nx': 720,  # 0.5度分辨率
                'ny': 361,  # 0.5度分辨率
                'nz': 40,
                'dx': 55000,  # 约55km
                'dy': 55000,
                'west_lon': -180.0,
                'east_lon': 180.0,
                'south_lat': -90.0,
                'north_lat': 90.0,
                'ref_lon': 0.0,
                'ref_lat': 0.0
            },
            'projection': {'type': 'latlon'}
        },

        'global_fine': {
            'domain': {
                'nx': 1440,  # 0.25度分辨率
                'ny': 721,  # 0.25度分辨率
                'nz': 50,
                'dx': 25000,  # 约25km
                'dy': 25000,
                'west_lon': -180.0,
                'east_lon': 180.0,
                'south_lat': -90.0,
                'north_lat': 90.0,
                'ref_lon': 0.0,
                'ref_lat': 0.0
            },
            'projection': {'type': 'latlon'}
        },

        'china': {
            'domain': {
                'nx': 280,  # 中国区域
                'ny': 160,
                'nz': 50,
                'dx': 25000,  # 25km
                'dy': 25000,
                'west_lon': 70.0,
                'east_lon': 140.0,
                'south_lat': 15.0,
                'north_lat': 55.0,
                'ref_lon': 105.0,
                'ref_lat': 35.0
            },
            'projection': {
                'type': 'lambert',
                'ref_lat': 35.0,
                'ref_lon': 105.0,
                'truelat1': 25.0,
                'truelat2': 45.0
            }
        }
    }

    # 选择配置（可以更改这里选择不同的配置）
    config_name = 'global_coarse'  # 改为 'global_medium', 'global_fine', 或 'china'
    config = configs[config_name]

    print(f"Using configuration: {config_name}")

    # 验证配置
    validate_config(config)

    # 创建处理器
    processor = ERA5GridProcessor(config)

    # 创建测试数据或使用实际ERA5数据
    era5_file = f"test_era5_data_{config_name}.nc"
    output_file = f"wrf_initialized_grid_{config_name}.nc"

    try:
        # 创建测试数据
        create_test_era5_data(era5_file)

        # 处理数据
        result = processor.process_era5_to_wrf_grid(era5_file, output_file)

        # 可视化结果
        if config_name.startswith('global'):
            processor.plot_global_results(result)
        else:
            processor.plot_results(result)

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()


def validate_config(config):
    """验证配置参数的合理性"""
    domain = config['domain']

    # 检查域大小
    lon_span = domain['east_lon'] - domain['west_lon']
    lat_span = domain['north_lat'] - domain['south_lat']

    print(f"Domain span: {lon_span:.1f}° × {lat_span:.1f}°")
    print(f"Approximate size: {lon_span * 111:.0f}km × {lat_span * 111:.0f}km")

    # 检查网格数量
    total_points = domain['nx'] * domain['ny'] * domain['nz']
    print(f"Total grid points: {total_points:,}")

    # 估算内存需求
    memory_gb = total_points * 10 * 8 / (1024 ** 3)  # 10个变量，8字节
    print(f"Estimated memory: {memory_gb:.1f} GB")

    # 全球网格检查
    is_global = (domain['west_lon'] == -180.0 and domain['east_lon'] == 180.0 and
                 domain['south_lat'] == -90.0 and domain['north_lat'] == 90.0)

    if is_global:
        resolution = 360 / domain['nx']
        print(f"Global grid resolution: {resolution:.3f}°")

    return True
    # 创建测试数据或使用实际ERA5数据
    era5_file = "era5_day_2021-01-01.nc"
    output_file = "wrf_initialized_grid.nc"

    # 如果没有实际ERA5数据，创建测试数据
    try:
        # 尝试使用实际的ERA5文件
        # era5_file = "path/to/your/actual/era5_data.nc"  # 取消注释使用实际数据
        create_test_era5_data(era5_file)  # 创建测试数据

        # 处理数据
        result = processor.process_era5_to_wrf_grid(era5_file, output_file)

        # 可视化结果
        processor.plot_results(result)

        print("Processing completed successfully!")

    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

    print("ERA5 to WRF grid initialization setup completed!")
    print("To use this code with real ERA5 data:")
    print("1. Install required packages: xarray, scipy, numpy, matplotlib")
    print("2. Download ERA5 data from Copernicus Climate Data Store")
    print("3. Update era5_file path to point to your actual ERA5 file")
    print("4. Run the processing")


if __name__ == "__main__":
    main()