# -*- coding: utf-8 -*-
"""
@author: sharrm

Created to experiment with image filtering techniques
"""

# open source
import numpy as np
from scipy.ndimage.filters import uniform_filter
from scipy import ndimage
from skimage import filters


class Filters:
    
    def median_filter(self, band, size):
        
        if type(band) == 'Array of complex128':
            print('Array cannot be complex to perform median filter')
        else:
            median_filtered = ndimage.fourier_ellipsoid(band, size=size)
            return median_filtered
        
    def hpf(self, original_band, fftshifted, size):
        
        rows, cols = original_band.shape
        crow,ccol = int(rows/2), int(cols/2)

        # remove the low frequencies by masking with a rectangular window of size 60x60
        # High Pass Filter (HPF)      
        fftshifted[crow-size:crow+size, ccol-size:ccol+size] = 0
        
        return fftshifted
    
    def fourier_ellipse(self, fft, size):
        fe_filtered = ndimage.fourier_ellipsoid(fft, size=size)
        
        return fe_filtered
    
    def fourier_gaussian(self, fft, sigma):
        
        fourier_gauss = ndimage.fourier_gaussian(fft, sigma=sigma)
        
        return fourier_gauss
    
    def uniform(self, band):
        
        pass
        
        