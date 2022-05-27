# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

import sys
# import glob
import os
import numpy as np
from scipy import fftpack
# import matplotlib.pyplot as plt
# from osgeo import gdal
# import rasterio as rio
# from scipy import ndimage, misc
# from scipy.ndimage.filters import uniform_filter
# from scipy.ndimage import correlate
# from skimage import filters
# from skimage.segmentation import active_contour, felzenszwalb, slic


class Fourier:
    
    def freqshifted(self, fft):
    
        # forward fft
        fftshifted = np.fft.fftshift(fft)
        
        return fftshifted
    
    def freq(self, band):
        
        fft = np.fft.fft2(band)
        
        return fft
    
    def magnitude_spectrum(self, band):
        fft = self.freq(band)
        fftshifted = self.freqshifted(fft)
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
    
    def fftpack(self, band):
        
        im_fft = fftpack.fft2(band.astype(float))
        
        return im_fft
    
    def ifftpack(self, band):
        
        im_fft = fftpack.ifft2(band)
        
        return im_fft