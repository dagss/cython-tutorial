import Image
import numpy as np
from matplotlib import pyplot as plt
import warnings
import contextlib
import time

# Utilities
def format_time(t):
    if t > 1 or t == 0:
        units = 's'
    elif t > 1e-3:
        units = 'ms'
        t *= 1e3
    elif t > 1e-6:
        units = 'us'
        t *= 1e6
    else:
        units = 'ns'
        t *= 1e9
    return '%.1f %s' % (t, units)

@contextlib.contextmanager
def take_time(desc):
    t0 = time.time()
    yield
    dt = time.time() - t0
    print '%s took %s' % (desc, format_time(dt))

def plot_images(images):
    if not isinstance(images, list):
        images = [images]
    fig, axs = plt.subplots(1, len(images))
    if len(images) == 1:
        axs = [axs] # ...
    for img, ax in zip(images, axs):
        i = ax.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
        if img.ndim == 2:
            fig.colorbar(i)

def color_convolve(convolve_func, img, kernel):
    smoothed_img = np.zeros((img.shape[0] + kernel.shape[0] - 1,
                             img.shape[1] + kernel.shape[1] - 1, 3),
                            np.float32)
    for cidx in range(3):
        smoothed_img[:, :, cidx] = convolve_func(img[:, :, cidx], kernel)
    return smoothed_img

def make_gaussian_kernel(ksize):
    x, y = np.mgrid[-ksize:ksize + 1, -ksize:ksize + 1]
    kernel = np.exp(-0.25 * (x**2 + y**2))
    kernel /= np.sum(kernel)
    return kernel

def load_image(filename):
    img = np.asarray(Image.open(filename))
    img = img.astype(np.float32) / 256
    return img

# Load photo
photo_lores = load_image('../google-lores.jpg')

# Python convolution
import convolve

kernel = make_gaussian_kernel(4)

with take_time('Python convolution'):
    smoothed_photo_lores = color_convolve(
        convolve.convolve2d, photo_lores, kernel)

plot_images([photo_lores, smoothed_photo_lores])


