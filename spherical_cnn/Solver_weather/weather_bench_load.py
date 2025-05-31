import xarray as xr
import numpy as np

# Google Cloud Storage 上 Keisler22 提供的 ERA5 数据源（6h，64x32）
GCS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_with_poles_conservative.zarr"

# 目标时间点
TARGET_TIME = np.datetime64("2020-06-01T12:00:00")

# 需要读取的变量
VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "2m_temperature",
    "surface_pressure",
]

# 打开远程 zarr 数据集（自动通过 gcsfs 加载）
print("🔄 正在加载 ERA5 数据...")
ds = xr.open_zarr(GCS_PATH, consolidated=True)

# 选取目标时间
print(f"📌 选取时间：{TARGET_TIME}")
ds_sel = ds.sel(time=TARGET_TIME)

# 显示变量维度信息
for var in VARIABLES:
    if var in ds_sel:
        da = ds_sel[var]
        print(f"\n✅ {var}: shape={da.shape}, dims={da.dims}")
    else:
        print(f"⚠️ 变量 {var} 不在数据集中")

output_path = "era5_20200601_12.nc"
ds_sel[VARIABLES].to_netcdf(output_path)
print(f"\n💾 已保存为 NetCDF 文件：{output_path}")