import xarray as xr
import numpy as np

# Google Cloud Storage 上 Keisler22 提供的 ERA5 数据源（6h，64x32）
GCS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-128x64_equiangular_with_poles_conservative.zarr"

# 目标时间点
TARGET_TIME = np.datetime64("2020-06-01T12:00:00")

# 需要读取的变量
VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "surface_pressure",
    "vertical_velocity",
    "specific_humidity"
]


ds = xr.open_zarr(GCS_PATH, consolidated=True)
ds_sel = ds.sel(time=TARGET_TIME)

# 显示变量维度信息
for var in VARIABLES:
    if var in ds_sel:
        da = ds_sel[var]
        print(f"{var}: shape={da.shape}, dims={da.dims}")
    else:
        print(f"变量 {var} 不在数据集中")

output_path = "era5_20200601_12.nc"
ds_sel[VARIABLES].to_netcdf(output_path)
print(f"已保存为 NetCDF 文件：{output_path}")
