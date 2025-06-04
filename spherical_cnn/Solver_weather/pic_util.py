import matplotlib.pyplot as plt
def draw(pic, lon, lat, scale = 1e6, title=""):

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)

    if lon is not None and lat is not None:
        # 假设 lon 和 lat 是 1D 数组
        extent = [lon.min(), lon.max(), lat.min(), lat.max()]
        plt.imshow(pic.T * scale, origin='lower', extent=extent, cmap='bwr', aspect='auto')
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
    else:
        plt.imshow(pic.T * scale, origin='lower', cmap='bwr', aspect='auto')

    plt.colorbar(label=title)
    plt.title(title)
    plt.tight_layout()
    plt.show()