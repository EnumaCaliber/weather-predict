import xarray as xr
import numpy as np
import random

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
years = np.arange(1959, 2023)
seasons = {
    "spring": [3, 4, 5],
    "summer": [6, 7, 8],
    "autumn": [9, 10, 11],
    "winter": [12, 1, 2]
}
sampled_years = random.sample(list(years), 25)
selected_pairs = []
# for year in sampled_years:
#     for season_months in seasons.values():
#         valid_times = [t for t in all_times
#                        if (np.datetime64(t).astype("datetime64[M]").astype(int) % 12 + 1) in season_months
#                        and (np.datetime64(t).astype("datetime64[Y]").astype(int) + 1970) == year]
#         if len(valid_times) >= 2:
#             t0 = random.choice(valid_times[:-1])
#             t1 = t0 + np.timedelta64(1, "h")
#             if t1 in all_times:
#                 selected_pairs.append((t0, t1))
#
# assert len(selected_pairs) == 100
attempts = 0

while len(selected_pairs) < 100 and attempts < 10000:
    year = random.choice(years)
    season_name, season_months = random.choice(list(seasons.items()))

    valid_times = [t for t in all_times
                   if (np.datetime64(t).astype("datetime64[M]").astype(int) % 12 + 1) in season_months
                   and (np.datetime64(t).astype("datetime64[Y]").astype(int) + 1970) == year]

    if len(valid_times) >= 2:
        t0 = random.choice(valid_times[:-1])
        t1 = t0 + np.timedelta64(1, "h")
        if t1 in all_times:
            selected_pairs.append((t0, t1))

    attempts += 1

assert len(selected_pairs) == 100, f"Only got {len(selected_pairs)} valid samples"

# 下载数据（共 200 个时间点）
time_points = sorted(set(t for pair in selected_pairs for t in pair))
ds_subset = ds.sel(time=time_points)[VARIABLES]

# 保存为 NetCDF 文件
ds_subset.to_netcdf("era5_100_dudt_samples.nc")
print("已保存为 era5_100_dudt_samples.nc")