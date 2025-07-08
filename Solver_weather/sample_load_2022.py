import xarray as xr
import numpy as np

# ERA5 数据 Zarr 云路径
GCS_PATH = "gs://weatherbench2/datasets/era5/1959-2022-1h-240x121_equiangular_with_poles_conservative.zarr/"
VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "surface_pressure",
    "vertical_velocity",
    "specific_humidity",
    "geopotential",
    "mean_sea_level_pressure",
    "2m_temperature"
]


ds = xr.open_zarr(GCS_PATH, consolidated=True)
all_times = ds.time.values
day = np.datetime64("2021-01-01")
one_day_times = [day + np.timedelta64(h, "h") for h in range(24)]
assert all(t in all_times for t in one_day_times), "miss some data"
ds_day = ds.sel(time=one_day_times)[VARIABLES]
filename = "era5_day_2021-01-01.nc"
ds_day.to_netcdf(filename)

print("Finish loading")