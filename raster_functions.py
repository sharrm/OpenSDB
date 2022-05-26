# -*- coding: utf-8 -*-
"""
@author: sharrm

Created for general functions
"""

import rasterio
import os

class RasterFunctions:
    
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
        