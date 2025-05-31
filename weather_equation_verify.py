import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


R_earth = 6.371e6
Omega = 7.2921e-5
R = 287.0


ds = xr.open_dataset("weather.nc")


t_idx = 0


u = ds["10m_u_component_of_wind"].values
v = ds["10m_v_component_of_wind"].values
T = ds["2m_temperature"].values
p = ds["surface_pressure"].values


lon = ds["longitude"].values  # shape: (lon,)
lat = ds["latitude"].values  # shape: (lat,)
lon2d, lat2d = np.meshgrid(lat, lon, indexing='xy')


phi = np.deg2rad(lat2d)
lambda_ = np.deg2rad(lon2d)

rho = p / (R * T)  # shape: (lat, lon)


f = 2 * Omega * np.sin(phi)



dlat = np.deg2rad(np.gradient(lat))  # shape: (lat,)
dlon = np.deg2rad(np.gradient(lon))  # shape: (lon,)


dphi = np.gradient(phi, axis=0)
dlambda = np.gradient(lambda_, axis=1)


du_dt = np.zeros_like(u)


dlon = np.deg2rad(np.gradient(lon))
dlon_2d = np.tile(dlon[:,np.newaxis], (1, 32))
dp_dlambda = np.gradient(p, axis=1) / dlon_2d

rhs = -1 / (rho * R_earth * np.cos(phi)) * dp_dlambda + f * v


lhs = -u * v * np.tan(phi) / R_earth


error = lhs - rhs


plt.figure(figsize=(6, 5))
plt.contourf(lat, lon, error, levels=21, cmap="bwr")
plt.title("$(LHS - RHS)/|RHS|$")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.colorbar()
plt.tight_layout()
plt.show()
