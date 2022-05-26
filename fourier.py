# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

import sys
# import glob
import os
import numpy as np
# import matplotlib.pyplot as plt
# from osgeo import gdal
# import rasterio as rio
# from scipy import ndimage, misc
# from scipy.ndimage.filters import uniform_filter
# from scipy.ndimage import correlate
# from skimage import filters
# from skimage.segmentation import active_contour, felzenszwalb, slic


class Fourier:
    
    def forward(self, band):
    
        # forward fft
        fftshifted = np.fft.fftshift(np.fft.fft2(band))
        
        return fftshifted
    
    def magnitude_spectrum(self, band):
        fftshifted = self.forward(band)
        magnitude = 20 * np.log(abs(fftshifted))
        # magnitude = np.fft.fft2(relative)

        return magnitude

    def inverse(self, band):
        
        fftshifted = self.forward(band)
        
        # inverse fft
        inverse = np.fft.ifftshift(np.fft.ifft2(fftshifted))
        # # f = np.fft.ifft2(F).real # could be useful in the future
        # real_inverse = abs(np.fft.ifft2(F))
        real_inverse = abs(inverse)
        
        return real_inverse
    