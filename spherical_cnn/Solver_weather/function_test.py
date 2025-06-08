import numpy as np
import xarray as xr


file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)

print(ds["level"])