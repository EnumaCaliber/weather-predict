import xarray as xr
from spherical_cnn.Solver_weather.weather_util import get_point_parameters

file_path = "era5_100_dudt_samples.nc"
ds = xr.open_dataset(file_path)
time = ds.time.values
level = 850
for time_index in range(0,100,2):
    time_curr = ds.time.values[time_index]
    time_next = ds.time.values[time_index + 1]
    ds_curr = ds.sel(time=time_curr)
    ds_next = ds.sel(time=time_next)
    util_curr = get_point_parameters(ds_curr)
    util_next = get_point_parameters(ds_next)

    u_curr = util_curr.get_wind_u(level=850)
    u_next = util_next.get_wind_u(level=850)


