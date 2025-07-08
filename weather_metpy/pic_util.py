import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings('ignore')


def draw(pic, lon=None, lat=None, scale=1e6, title="",
         cmap='bwr', figsize=(14, 5), save_path=None,
         show_contours=True, contour_levels=20,
         vmin=None, vmax=None, units="",
         projection=None, add_coastlines=False,
         subplot_position=None, fig=None, ax=None):
    """
    改进的绘图函数，支持多种绘图选项和错误处理

    Parameters:
    -----------
    pic : np.ndarray
        要绘制的数据数组 (2D)
    lon : np.ndarray, optional
        经度数组，可以是1D或2D
    lat : np.ndarray, optional
        纬度数组，可以是1D或2D
    scale : float, default=1e6
        数据缩放因子
    title : str, default=""
        图表标题
    cmap : str, default='bwr'
        颜色映射
    figsize : tuple, default=(14, 5)
        图表大小
    save_path : str, optional
        保存路径
    show_contours : bool, default=True
        是否显示等值线
    contour_levels : int, default=20
        等值线数量
    vmin, vmax : float, optional
        颜色范围
    units : str, default=""
        数据单位
    projection : str, optional
        地图投影 (需要cartopy)
    add_coastlines : bool, default=False
        是否添加海岸线 (需要cartopy)
    subplot_position : tuple, optional
        子图位置 (rows, cols, index)
    fig, ax : matplotlib objects, optional
        现有的图表对象

    Returns:
    --------
    fig, ax : matplotlib objects
        图表对象
    """

    # 数据验证
    if not isinstance(pic, np.ndarray):
        raise TypeError("pic must be a numpy array")

    if pic.ndim != 2:
        raise ValueError("pic must be a 2D array")

    if np.all(np.isnan(pic)):
        raise ValueError("pic contains only NaN values")

    # 处理数据
    data = pic * scale

    # 处理无效值
    if np.any(np.isinf(data)) or np.any(np.isnan(data)):
        print("Warning: Data contains NaN or infinite values")
        data = np.nan_to_num(data, nan=0.0, posinf=np.nanmax(data[np.isfinite(data)]),
                             neginf=np.nanmin(data[np.isfinite(data)]))

    # 创建图表
    if fig is None or ax is None:
        if subplot_position is not None:
            fig, ax = plt.subplots(subplot_position[0], subplot_position[1], figsize=figsize)
            if isinstance(ax, np.ndarray):
                ax = ax.flatten()[subplot_position[2]]
        else:
            fig, ax = plt.subplots(figsize=figsize)

    # 处理坐标
    if lon is not None and lat is not None:
        lon, lat, data = _process_coordinates(lon, lat, data)

        # 使用contourf绘制
        if show_contours:
            if vmin is None:
                vmin = np.nanpercentile(data, 5)
            if vmax is None:
                vmax = np.nanpercentile(data, 95)

            levels = np.linspace(vmin, vmax, contour_levels)
            im = ax.contourf(lon, lat, data, levels=levels, cmap=cmap, extend='both')

            # 添加等值线
            contour_lines = ax.contour(lon, lat, data, levels=levels, colors='black',
                                       linewidths=0.5, alpha=0.6)
            ax.clabel(contour_lines, inline=True, fontsize=8, fmt='%.1e')
        else:
            im = ax.pcolormesh(lon, lat, data, cmap=cmap, vmin=vmin, vmax=vmax)

        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")

        # 设置坐标轴范围
        ax.set_xlim(np.nanmin(lon), np.nanmax(lon))
        ax.set_ylim(np.nanmin(lat), np.nanmax(lat))

    else:
        # 使用imshow绘制
        if vmin is None or vmax is None:
            vmin = np.nanpercentile(data, 5)
            vmax = np.nanpercentile(data, 95)

        im = ax.imshow(data.T, origin='lower', cmap=cmap,
                       aspect='auto', vmin=vmin, vmax=vmax)
        ax.set_xlabel("X Index")
        ax.set_ylabel("Y Index")

    # 添加颜色条
    cbar_label = f"{title} ({units})" if units else title
    cbar = plt.colorbar(im, ax=ax, label=cbar_label)

    # 设置标题
    ax.set_title(title, fontsize=12, fontweight='bold')

    # 添加网格
    ax.grid(True, alpha=0.3)

    # 添加统计信息
    _add_statistics_text(ax, data, scale)

    # 地图投影和海岸线 (需要cartopy)
    if projection or add_coastlines:
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            if projection:
                ax.set_projection(ccrs.PlateCarree())

            if add_coastlines:
                ax.add_feature(cfeature.COASTLINE)
                ax.add_feature(cfeature.BORDERS)

        except ImportError:
            print("Warning: cartopy not installed, skipping map features")

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.tight_layout()

    return fig, ax


def _process_coordinates(lon, lat, data):
    """处理坐标数组"""

    # 转换为numpy数组
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    # 处理1D坐标
    if lon.ndim == 1 and lat.ndim == 1:
        # 创建网格
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # 确保数据维度匹配
        if data.shape != lon_grid.shape:
            if data.shape == lon_grid.T.shape:
                data = data.T
            else:
                raise ValueError(f"Data shape {data.shape} doesn't match coordinate grid {lon_grid.shape}")

        return lon_grid, lat_grid, data

    # 处理2D坐标
    elif lon.ndim == 2 and lat.ndim == 2:
        # 检查形状匹配
        if lon.shape != lat.shape:
            raise ValueError("lon and lat must have the same shape")

        if data.shape != lon.shape:
            if data.shape == lon.T.shape:
                data = data.T
            else:
                raise ValueError(f"Data shape {data.shape} doesn't match coordinate shape {lon.shape}")

        return lon, lat, data

    else:
        raise ValueError("lon and lat must be either both 1D or both 2D")


def _add_statistics_text(ax, data, scale):
    """添加统计信息文本"""

    # 计算统计量
    valid_data = data[np.isfinite(data)]
    if len(valid_data) == 0:
        return

    stats = {
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'mean': np.mean(valid_data),
        'std': np.std(valid_data)
    }

    # 创建统计文本
    stats_text = f"Min: {stats['min']:.2e}\nMax: {stats['max']:.2e}\nMean: {stats['mean']:.2e}\nStd: {stats['std']:.2e}"

    # 添加到图表
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=8, fontfamily='monospace')


def draw_comparison(data_list, titles, lon=None, lat=None, scale=1e6,
                    figsize=(18, 12), cmap='bwr', save_path=None,
                    units="", suptitle=""):
    """
    绘制多个数据的对比图

    Parameters:
    -----------
    data_list : list of np.ndarray
        要对比的数据列表
    titles : list of str
        每个子图的标题
    其他参数同draw函数
    """

    n_plots = len(data_list)
    if n_plots != len(titles):
        raise ValueError("data_list and titles must have the same length")

    # 计算子图布局
    cols = min(3, n_plots)
    rows = (n_plots + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()

    # 计算全局颜色范围
    all_data = np.concatenate([d.flatten() * scale for d in data_list])
    vmin = np.nanpercentile(all_data, 5)
    vmax = np.nanpercentile(all_data, 95)

    # 绘制每个子图
    for i, (data, title) in enumerate(zip(data_list, titles)):
        if i < len(axes):
            draw(data, lon=lon, lat=lat, scale=scale, title=title,
                 cmap=cmap, vmin=vmin, vmax=vmax, units=units,
                 fig=fig, ax=axes[i])

    # 隐藏多余的子图
    for i in range(len(data_list), len(axes)):
        axes[i].axis('off')

    # 添加总标题
    if suptitle:
        fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison figure saved to {save_path}")

    plt.tight_layout()
    return fig, axes


def draw_with_vectors(scalar_field, u_field, v_field, lon=None, lat=None,
                      scale=1e6, title="", figsize=(14, 8), cmap='bwr',
                      vector_scale=1, vector_skip=3, save_path=None, units=""):
    """
    绘制标量场和矢量场的组合图

    Parameters:
    -----------
    scalar_field : np.ndarray
        标量场数据
    u_field, v_field : np.ndarray
        矢量场的u和v分量
    vector_scale : float, default=1
        矢量缩放因子
    vector_skip : int, default=3
        矢量绘制间隔
    """

    fig, ax = plt.subplots(figsize=figsize)

    # 绘制标量场
    fig, ax = draw(scalar_field, lon=lon, lat=lat, scale=scale, title=title,
                   cmap=cmap, fig=fig, ax=ax, units=units)

    # 添加矢量场
    if lon is not None and lat is not None:
        lon, lat, _ = _process_coordinates(lon, lat, scalar_field)

        # 下采样矢量
        lon_vec = lon[::vector_skip, ::vector_skip]
        lat_vec = lat[::vector_skip, ::vector_skip]
        u_vec = u_field[::vector_skip, ::vector_skip] * vector_scale
        v_vec = v_field[::vector_skip, ::vector_skip] * vector_scale

        # 绘制矢量
        ax.quiver(lon_vec, lat_vec, u_vec, v_vec,
                  color='black', alpha=0.7, scale_units='xy', scale=1)

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Vector field figure saved to {save_path}")

    plt.tight_layout()
    return fig, ax


# 使用示例
def example_usage():
    """使用示例"""

    # 创建示例数据
    nx, ny = 100, 80
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y)

    # 创建一些示例数据
    data1 = np.exp(-(X ** 2 + Y ** 2) / 10) * np.sin(X) * np.cos(Y)
    data2 = np.cos(X) * np.sin(Y) / (1 + X ** 2 + Y ** 2)

    # 创建坐标
    lon = np.linspace(-130, -60, nx)
    lat = np.linspace(20, 60, ny)

    print("Example 1: Basic usage")
    fig, ax = draw(data1, lon=lon, lat=lat, scale=1e6,
                   title="Example Data 1", units="×10⁻⁶")
    plt.show()

    print("\nExample 2: Comparison plot")
    fig, axes = draw_comparison([data1, data2],
                                ["Data 1", "Data 2"],
                                lon=lon, lat=lat, scale=1e6,
                                suptitle="Data Comparison", units="×10⁻⁶")
    plt.show()

    print("\nExample 3: Vector field")
    u_field = -np.gradient(data1, axis=1)
    v_field = -np.gradient(data1, axis=0)
    fig, ax = draw_with_vectors(data1, u_field, v_field,
                                lon=lon, lat=lat, scale=1e6,
                                title="Scalar + Vector Field",
                                vector_scale=1e6, units="×10⁻⁶")
    plt.show()


if __name__ == "__main__":
    example_usage()