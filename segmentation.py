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
        felensz = felzenszwalb(gradient, scale = 2, sigma=5, min_size=1000)
        
        return felensz