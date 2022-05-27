# -*- coding: utf-8 -*-
"""
@author: sharrm

Created to experiment with image segmentation techniques
"""

# open source
# from scipy import ndimage, misc
from scipy.ndimage import correlate
from skimage.segmentation import active_contour, felzenszwalb, slic


class Segmentation:
    
    def felensz(self, gradient, scale, sigma, min_size):
        felensz = felzenszwalb(gradient, scale = scale, sigma=sigma, min_size=min_size)
        
        return felensz