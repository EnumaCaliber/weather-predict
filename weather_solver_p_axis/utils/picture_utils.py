import matplotlib.pyplot as plt
import numpy as np


def draw_clean(pic, lon=None, lat=None, scale=1e6, title="",
               cmap='bwr', figsize=(14, 5), save_path=None,
               vmin=None, vmax=None, units="",
               fig=None, ax=None):


    if not isinstance(pic, np.ndarray):
        raise TypeError("pic must be a numpy array")

    if pic.ndim != 2:
        raise ValueError("pic must be a 2D array")

    if np.all(np.isnan(pic)):
        raise ValueError("pic contains only NaN values")


    data = pic * scale


    if np.any(np.isinf(data)) or np.any(np.isnan(data)):
        print("Warning: Data contains NaN or infinite values")
        data = np.nan_to_num(data, nan=0.0,
                             posinf=np.nanmax(data[np.isfinite(data)]),
                             neginf=np.nanmin(data[np.isfinite(data)]))

    # 创建图表
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # 处理坐标
    if lon is not None and lat is not None:
        lon, lat, data = _process_coordinates(lon, lat, data)

        # 设置颜色范围
        if vmin is None:
            vmin = np.nanpercentile(data, 5)
        if vmax is None:
            vmax = np.nanpercentile(data, 95)

        # Pure plotting: only pcolormesh, no contour lines
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
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')

    # Remove all decorations: no grid, no contour lines, no statistics

    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.tight_layout()
    plt.show()
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


# ===============================================
# 极简版本 - 最纯净的显示
# ===============================================

def draw_minimal(pic, lon=None, lat=None, cmap='bwr', figsize=(12, 8)):
    """
    Minimal plotting function - cleanest display

    Parameters:
    -----------
    pic : np.ndarray
        Data array to plot (2D)
    lon, lat : np.ndarray, optional
        Longitude and latitude coordinates
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    """

    fig, ax = plt.subplots(figsize=figsize)

    if lon is not None and lat is not None:
        # Process coordinates
        lon, lat, pic = _process_coordinates(lon, lat, pic)

        # Clean display
        im = ax.pcolormesh(lon, lat, pic, cmap=cmap)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
    else:
        # Display without coordinates
        im = ax.imshow(pic.T, origin='lower', cmap=cmap, aspect='auto')

    # Only add colorbar
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()
    return fig, ax


# ===============================================
# 批量显示版本
# ===============================================

def draw_multiple_clean(data_list, titles=None, lon=None, lat=None,
                        cmap='bwr', figsize=(15, 10), cols=3):
    """
    Batch clean display of multiple data

    Parameters:
    -----------
    data_list : list of np.ndarray
        List of data arrays
    titles : list of str, optional
        List of titles
    lon, lat : np.ndarray, optional
        Longitude and latitude coordinates
    cmap : str
        Colormap
    figsize : tuple
        Figure size
    cols : int
        Number of columns
    """

    n_data = len(data_list)
    rows = (n_data + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Handle axes for 1D case
    if rows == 1:
        axes = axes.reshape(1, -1) if n_data > 1 else [axes]
    elif cols == 1:
        axes = axes.reshape(-1, 1)

    # Ensure axes is 2D array
    if n_data == 1:
        axes = np.array([[axes]])

    # Calculate global color range
    all_data = np.concatenate([data.flatten() for data in data_list])
    vmin = np.nanpercentile(all_data, 5)
    vmax = np.nanpercentile(all_data, 95)

    for i, data in enumerate(data_list):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 or cols > 1 else axes[0]

        if lon is not None and lat is not None:
            lon_proc, lat_proc, data_proc = _process_coordinates(lon, lat, data)
            im = ax.pcolormesh(lon_proc, lat_proc, data_proc,
                               cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        else:
            im = ax.imshow(data.T, origin='lower', cmap=cmap,
                           aspect='auto', vmin=vmin, vmax=vmax)

        # Add title
        if titles and i < len(titles):
            ax.set_title(titles[i])

        plt.colorbar(im, ax=ax)

    # Hide extra subplots
    for i in range(n_data, rows * cols):
        row = i // cols
        col = i % cols
        if rows > 1 or cols > 1:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.show()
    return fig, axes


# ===============================================
# 使用示例
# ===============================================

def example_clean_plotting():
    """Clean plotting examples"""

    # Create sample data
    nx, ny = 100, 80
    x = np.linspace(-10, 10, nx)
    y = np.linspace(-5, 5, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    data1 = np.exp(-(X ** 2 + Y ** 2) / 10) * np.sin(X) * np.cos(Y)
    data2 = np.cos(X) * np.sin(Y) / (1 + X ** 2 + Y ** 2)

    # Simulate longitude and latitude
    lon = np.linspace(-130, -60, nx)
    lat = np.linspace(20, 60, ny)

    print("Example 1: Clean single plot display")
    fig1, ax1 = draw_clean(data1, lon=lon, lat=lat,
                           title="Clean Plot Example")

    print("Example 2: Minimal display")
    fig2, ax2 = draw_minimal(data1, lon=lon, lat=lat)

    print("Example 3: Batch clean display")
    fig3, axes3 = draw_multiple_clean([data1, data2],
                                      titles=["Data 1", "Data 2"],
                                      lon=lon, lat=lat)


if __name__ == "__main__":
    example_clean_plotting()