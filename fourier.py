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
    
    def freqshifted(self, band):
    
        # forward fft
        fftshifted = np.fft.fftshift(np.fft.fft2(band))
        
        return fftshifted
    
    def freq(self, band):
        
        fft = np.fft.fft2(band)
        
        return fft
    
    def magnitude_spectrum(self, band):
        fftshifted = self.freqshifted(band)
        magnitude = 20 * np.log(abs(fftshifted))
        # magnitude = np.fft.fft2(relative)

        return magnitude

    def inverse(self, fftshifted):
        
        # fftshifted = self.freqshifted(band)
        inverse = np.fft.ifft2(fftshifted)
        # # f = np.fft.ifft2(F).real # could be useful in the future
        # real_inverse = abs(np.fft.ifft2(F))
        
        return inverse
    
    def inverse_shifted(self, fftshifted):
        
        # fftshifted = self.freqshifted(band)
        inverseshifted = np.fft.ifft2(np.fft.ifftshift(fftshifted))
        # # f = np.fft.ifft2(F).real # could be useful in the future
        # real_inverse = abs(np.fft.ifft2(F))
        
        return inverseshifted
    