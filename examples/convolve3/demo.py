import Image
import numpy as np
from matplotlib import pyplot as plt
import warnings
import contextlib
import time
import threading

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

    def thread_main(cidx):
        color_image = img[:, :, cidx].copy()
        smoothed_img[:, :, cidx] = convolve_func(color_image, kernel)

    threads = [threading.Thread(target=thread_main, args=(cidx,))
               for cidx in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return smoothed_img

def make_gaussian_kernel(ksize):
    x, y = np.mgrid[-ksize:ksize + 1, -ksize:ksize + 1]
    kernel = np.exp(-0.25 * (x**2 + y**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)

def load_image(filename):
    img = np.asarray(Image.open(filename))
    img = img.astype(np.float32) / 256
    return img

# Low-resolution data
photo_lores = load_image('../google-lores.jpg')
kernel = make_gaussian_kernel(4)

# Python convolution
import convolve
with take_time('Python convolution'):
    smoothed_photo = color_convolve(convolve.convolve2d,
                                    photo_lores, kernel)

# Cython convolution
import cy_convolve
with take_time('Cython convolution'):
    smoothed_photo = color_convolve(cy_convolve.convolve2d,
                                    photo_lores, kernel)

# Move to hi-resolution to better measure effects on Cython code
photo_hires = load_image('../google.jpg')
kernel = make_gaussian_kernel(10)

with take_time('Cython convolution (hi-res)'):
    smoothed_photo = color_convolve(cy_convolve.convolve2d,
                                    photo_hires, kernel)

with take_time('Tuned Cython convolution (hi-res)'):
    smoothed_photo = color_convolve(cy_convolve.convolve2d_tuned,
                                    photo_hires, kernel)

plot_images([photo_hires, smoothed_photo])

plt.show()
