# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 12:31:18 2022

@author: sharrm
"""

import radialProfile
import glob
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import rasterio

# rasters = []

# for tif in glob.glob(r'P:\SDB\Test_Files\Samples\Rasters\*.tif', recursive=False):
#     rasters.append(tif)

# rasters.sort()

# psd1d, psd2d, img = radialProfile.PSD(rasters)

## create experiment array
a = np.ones([11,11])
a[1:2,1:10] = 10
a[3:4,1:10] = 10
a[5:6,1:10] = 10
a[7:8,1:10] = 10
a[9:10,1:10] = 10

# pad with zeros
a = np.pad(a, ((1, 1), (1, 1)), 'constant', constant_values=0)

tif = r"P:\SDB\Test_Files\Noise_Files\MoreHalf.tif"
a = rasterio.open(tif).read(1)
a[a == -9999] = np.nan

# pad with zeros
a = np.pad(a, ((1, 2), (1, 2)), 'constant', constant_values=(np.nan))

rows, cols = a.shape

b = np.zeros([rows, cols])

# i = 100
# j = 100

# window = a[i-4:i+5, j-4:j+5]
# trapz = radialProfile.PSD(window)

wsize = 4

for i in np.arange(wsize, rows-wsize, 1):
    for j in np.arange(wsize, cols-wsize, 1):
        window = a[i-wsize : i+wsize+1, j-wsize : j+wsize+1]
        # print(f' {i},{j}: \n{window}')
        # b[i,j] = np.mean(window)
        b[i,j] = radialProfile.PSD(window)
        print(f'Location {i},{j}: {b[i,j]}')
        
# https://stackoverflow.com/questions/15722324/sliding-window-of-m-by-n-shape-numpy-ndarray
# f = a.flatten()

# indexer = np.arange(6)[None, :] + 2*np.arange(4)[:, None]

# psd1d, psd2d, trapz = radialProfile.PSD(a)

# plt.imshow(psd2d)
# plt.show()


# b = np.array([[-1, -1, -1],
#               [-1, 10, -1],
#               [-1, -1, -1]])

# c = ndimage.convolve(a, b)



