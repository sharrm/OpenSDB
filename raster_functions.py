# -*- coding: utf-8 -*-
"""
@author: sharrm

Created for general functions
"""


import numpy as np
import os
import rasterio
from scipy.ndimage.filters import uniform_filter


class ReadWriteFunctions:
    
    # read image using rasterio and return rasterio object
    def read_image(self, band):
        # read geotiff with rasterio
        image_open = rasterio.open(band)
                
        return image_open

    # build array from rasterio object (single band) and return it
    def image_array(self, band):
        # read band
        try:
            image_read = self.read_image(band).read(1)
        except:
            print('May be using an image with more than one band.')
        
        return image_read
        
    # store useful metadata from image
    def meta_data(self, band):
        
        image = self.read_image(band)
        meta = image.meta
        
        return meta
    
    def write_raster(self, path, name, metadata, array):
        
        with rasterio.open(os.path.join(path, name), "w", **metadata) as dest:
            dest.write(array, 1) # write array to raster
        
class ArrayOperations:
    
    def gradient(self, band):
        gradient = np.gradient(band, axis=0)
        
        return gradient
        
    def window_stdev(arr, radius):
        c1 = uniform_filter(arr, radius*2, mode='constant', origin=-radius)
        c2 = uniform_filter(arr*arr, radius*2, mode='constant', origin=-radius)
        
        std = ((c2 - c1*c1)**.5)[:-radius*2+1,:-radius*2+1]
        
        return std
    
    def mask(self, in_array, in_mask, nodata):
        
        masked = np.where(in_mask == 1, in_mask, in_array)
        masked = np.where(masked == 1, nodata, in_array)
        
        return masked
        
    
    # def std(self, band):
        
        