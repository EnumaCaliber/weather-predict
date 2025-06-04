import matplotlib.pyplot as plt
def draw(pic, title=""):
    scale = 1e6
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pic.T * scale, origin='lower', cmap='bwr')
    plt.colorbar(label=title)
    plt.title(title)

    plt.tight_layout()
    plt.show()