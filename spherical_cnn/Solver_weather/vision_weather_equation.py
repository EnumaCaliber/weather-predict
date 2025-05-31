import h5py
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

filename = "snapshots/snapshots_s1.h5"
with h5py.File(filename, 'r') as file:
    height = file['tasks']['height'][:]        # shape = (T, 1, Ny, Nx)
    phi = file['scales']['phi']['1.0'][:]
    theta = file['scales']['theta']['1.0'][:]

lat = 90 - np.rad2deg(theta)
lon = np.rad2deg(phi)
Lon, Lat = np.meshgrid(lon, lat)

# 画多个时间步图像
for i in range(height.shape[0]):
    fig = plt.figure(figsize=(8, 4))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cf = ax.contourf(Lon, Lat, height[i,0], 60, cmap='viridis', transform=ccrs.PlateCarree())
    ax.coastlines()
    plt.colorbar(cf, label='Height')
    plt.title(f"Time step {i}")
    plt.savefig(f"frame_{i:03d}.png", dpi=150)
    plt.close()