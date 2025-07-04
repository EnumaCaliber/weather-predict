import xarray as xr

from spherical_cnn.Solver_weather.random_sample_test_du_dt import level

file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)


print(ds.sel(level=850)["vertical_velocity"].values )