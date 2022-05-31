# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.mask
# from rasterio import plot
# from rasterio.plot import show
import fiona
#import geopandas as gpd
# from osgeo import gdal


def mask_imagery(red, green, blue, in_shp):

    # list of bands
    rasters = [blue, green, red]

    # dict to store output file names
    masked_rasters = {}

    #open bounding shapefile
    with fiona.open(in_shp, 'r') as shapefile:
        shape = [feature['geometry'] for feature in shapefile]

    raster_list = []
    #loop through input rasters
    for band in rasters:
        # read raster, extract spatial information, mask the raster using the input shapefile
        with rasterio.open(band) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
            # nodata = src.nodata

        # writing information
        out_meta.update({"driver": "GTiff",
                         # "dtype": 'float32',
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                          "transform": out_transform},)

        # simply customizing the output filenames here -- there's probably a better method
        if '492' in band or 'B2' in band: # blue wavelength (492nm)
            outraster_name = os.path.join(os.path.dirname(band), 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['blue'] = outraster_name
            print(out_image)
        elif '560' in band or 'B3' in band: # green wavelength (560nm)
            outraster_name = os.path.join(os.path.dirname(band), 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['green'] = outraster_name
        elif '665' in band or 'B4' in band: # red wavelength (665nm)
            outraster_name = os.path.join(os.path.dirname(band), 'masked_' + os.path.basename(band)[-7:-4] + '.tif')
            masked_rasters['red'] = outraster_name

        # write masked raster to a file
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(out_image)

        raster_list.append(outraster_name)
        # close the file
        dest = None

    return True, raster_list


def pSDBgreen (blue, green, in_shp, rol_name):

    # read blue band
    with rasterio.open(blue) as blue_src:
        blue_image = blue_src.read(1)

    # read green band
    with rasterio.open(green) as green_src:
        green_image = green_src.read(1)
        out_meta = green_src.meta

    # increase band values by factor of 1,000
    ratioBlueArrayOutput = blue_image * 1000.0
    ratioGreenArrayOutput = green_image * 1000.0
    
    print(type(ratioBlueArrayOutput))
    print(ratioBlueArrayOutput)

    # calculate natural log of each band
    lnBlueArrayOutput = np.log(ratioBlueArrayOutput)
    lnGreenArrayOutput = np.log(ratioGreenArrayOutput)

    # compute ratio between bands
    ratioImage = lnBlueArrayOutput / lnGreenArrayOutput
    
    # output raster filename with path
    outraster_name = os.path.join(os.path.dirname(blue), rol_name + '.tif')
    
    # writing information  
    ratioImage[np.isnan(ratioImage)] = -9999.0
    ratioImage[np.isinf(ratioImage)] = -9999.0
    out_meta.update({"dtype": 'float32', "nodata": -9999.0})
    
    # write ratio between bands to a file
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(ratioImage, 1)

    # close the file
    dest = None

    print(f"The ratio raster file is called: {outraster_name}")

    return True, outraster_name, blue_image, ratioBlueArrayOutput, lnBlueArrayOutput, ratioImage


# def read_band(band):
#     with rasterio.open(masked_rasters['blue']) as src:
#         read_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
#         out_meta = src.meta

#     return read_image, out_meta, out_transform

# def relative_bathymetry(band1, band2):
#     band1, ref, transform = read_band(band1)
#     band2, ref, transform = read_band(band2)

#     # Stumpf algorithm
#     ratiologs = np.log(1000 * band1) / np.log(1000 * band2)

#     return ratiologs, ref, transform

# def write_raster(band1, band2):
#     output_rol = relative_bathymetry(band1, band2)

#     # output raster filename with path
#     outraster_name = os.path.join(os.path.dirname(band1), 'ratio_of_logs.tif')

#     # write ratio between bands to a file
#     with rasterio.open(outraster_name, "w", **out_meta) as dest:
#         dest.write(ratioImage)

#     # close the file
#     dest = None

#     return None
