# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

if __name__ == '__main__':
    
    # import cv2
    import os
    import math
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import rasterio
    import rasterio.mask
    import fiona
    import glob
    
    # in_shp = r"P:\SDB\Test_Files\clipper.shp"
    band = r"P:\SDB\Test_Files\KeyLargoROL_Full.tif"
    
    shapefiles = []
    rasters = []

    for shp in glob.glob(r'P:\SDB\Test_Files\ClipFiles\**\*.shp', recursive=True):
        if shp not in shapefiles:
            shapefiles.append(shp)
      
    for shp in shapefiles:
    
        with fiona.open(shp, 'r') as shapefile:
            shape = [feature['geometry'] for feature in shapefile]
            
    #         # read raster, extract spatial information, mask the raster using the input shapefile
        with rasterio.open(band) as src:
            out_image, out_transform = rasterio.mask.mask(src, shape, crop=True)
            out_meta = src.meta
            
    #     # writing information
        out_meta.update({"driver": "GTiff",
                          "height": out_image.shape[1],
                          "width": out_image.shape[2],
                          "transform": out_transform})
    
        outraster_name = os.path.join(r'P:\SDB\Test_Files\Noise_Files', os.path.basename(shp)[:-4] + '.tif')
        # print(outraster_name)

        # write masked raster to a file
        with rasterio.open(outraster_name, "w", **out_meta) as dest:
            dest.write(out_image)
            
        rasters.append(outraster_name)
        print(f'Wrote raster here: {outraster_name}')
    
    # for tif in glob.glob(r'P:\SDB\Test_Files\Noise_Files\*.tif', recursive=False):
    #     rasters.append(tif)
        
    # print(rasters)
    
    # for tif in rasters:
        
    #     tifname = os.path.basename(tif)
        
    #     if tifname == 'Noiseless.tif' or tifname == 'Full.tif': # or tifname == 'AlmostHalf.tif'
        
    #         img = rasterio.open(tif)
    #         img = img.read(1)
            
    #         # print( img.shape )
            
    #         # Fourier Transform along the first axis
            
    #         # Round up the size along this axis to an even number
    #         n = int( math.ceil(img.shape[0] / 2.) * 2 )
            
    #         # Use rfft since we are processing real values
    #         y = np.fft.rfft(img,n, axis=0)
            
    #         # Sum power along the second axis
    #         y = y.real*y.real + y.imag*y.imag
    #         y = y.sum(axis=1)/y.shape[1]
            
    #         # Generate a list of frequencies
    #         freqy = np.fft.rfftfreq(n)
            
    #         # Graph it
    #         # plt.plot(freqy[1:],y[1:], label = f'amp y vs f_x - {tifname} ')
            
    #         # Fourier Transform along the second axis
            
    #         # Same steps as above
    #         n = int( math.ceil(img.shape[1] / 2.) * 2 )
            
    #         x = np.fft.rfft(img,n,axis=1)
            
    #         x = x.real*x.real + x.imag*x.imag
    #         x = x.sum(axis=0)/x.shape[0]
            
    #         freqx = np.fft.rfftfreq(n)
            
    #         # plt.plot(freqx[1:],x[1:],  label = f'amp x vs f_y - {tifname}')
            
    #         averagefreq = np.mean([freqx, freqy], axis=0)
    #         average = np.mean([x, y], axis=0)
            
    #         plt.plot(averagefreq[1:],average[1:],  label = f'Average - {tifname}')
            
    #         plt.ylabel( 'Amplitude' )
    #         plt.xlabel( 'Frequency' )
    #         plt.yscale( 'log' )
            
    #         plt.title(os.path.basename(tif))
    #         plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0.05),
    #       ncol=3, fancybox=True)
    #         plt.grid()
            
    #         # plt.show()
    #         # plt.savefig(os.path.join(r'P:\SDB\Test_Files\Noise_Files\Figures', os.path.basename(tif)[:-4] + '.png'), dpi=300)
            
    #         img = None
        
    #     # plt.savefig(os.path.join(r'P:\SDB\Test_Files\Noise_Files\Figures', 'Comparison.png'), dpi=300)
        
    # plt.show()
        
    
    
    
    
    
    # import sys
    # import numpy as np
    # # import matplotlib.pyplot as plt
    # # from osgeo import gdal
    # import rasterio as rio
    # # from scipy import ndimage, misc
    # from scipy.ndimage.filters import uniform_filter
    # from scipy.ndimage import correlate
    # from skimage import filters
    # from skimage.segmentation import active_contour, felzenszwalb, slic
    # # import glob
    # import os
    
    # importing from tutorial
    # https://towardsdatascience.com/image-processing-with-python-application-of-fourier-transformation-5a8584dc175b
    # from skimage.io import imread, imshow
    # from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
    # from skimage import color, exposure, transform
    # from skimage.exposure import equalize_hist
    
    # check if not using Covelli version as interpreter
    # if not 'Covelli' in sys.executable:
    #     print(f'Using Python interpreter: {sys.executable}')
    # else:
    #     print(f'\n* Using interpreter: {sys.executable}\n')
    
    # band = r"P:\SDB\Florida Keys\Popcorn\Test_Files\rel_test.tif" # masked in arc
    # band = r"P:\SDB\Florida Keys\Popcorn\Test_Files\masked.tif"
    # band = r"U:\ce567\sharrm\Lab7\dataset_files\switched_SDB.tif"
    
    # print(f'Processing: {band}')
    
    # read geotiff with rasterio and mask no data
    # dark_image = rio.open(band)
    # relative = dark_image.read(1)
    # out_meta = dark_image.meta
    # out_transform = dark_image.transform
    # nodata = dark_image.nodata
    # relative = np.where(relative == nodata, 0, relative)
    
    # # gdal version
    # # dark_image = gdal.Open(band)
    # # relative = dark_image.GetRasterBand(1).ReadAsArray()
    
    # print(f'\nRead image band: {band}')
    
    # # forward fft
    # relative_fft = np.fft.fftshift(np.fft.fft2(relative))
    # magnitude = 20 * np.log(abs(relative_fft))
    # # magnitude = np.fft.fft2(relative)
    
    # # gaussian fourier filter
    # # gaussfourier = ndimage.fourier_gaussian(magnitude, sigma=4)
    # # inverse = np.fft.ifft2(gaussfourier)
    
    
    # # inverse fft
    # to_inverse = relative_fft # being lazy -- to more easily change the inverse
    # F = np.fft.ifftshift(to_inverse)
    # # # f = np.fft.ifft2(F).real # could be useful in the future
    # inverse = abs(np.fft.ifft2(F))
    
    # # difference between input and result
    # check = np.subtract(relative, inverse)
    
    # # plotting
    # fig, ax = plt.subplots(1,3,figsize=(15,15))
    # ax[0].imshow(relative, cmap='gray')
    # ax[0].set_title('Original')
    # ax[1].imshow(magnitude.real, cmap='gray')
    # ax[1].set_title('Magnitude')
    # ax[2].imshow(inverse, cmap='gray')
    # ax[2].set_title('iFFT')
    # plt.show()
    
    
    # slope stuff
    
    # def window_stdev(arr, radius):
    #     c1 = uniform_filter(arr, radius*2, mode='constant', origin=-radius)
    #     c2 = uniform_filter(arr*arr, radius*2, mode='constant', origin=-radius)
    #     return ((c2 - c1*c1)**.5)[:-radius*2+1,:-radius*2+1]
    
    # # print(f'Relative shape: {relative.shape}')
    # gradient = np.gradient(relative, axis=0).astype('float64')
    # std = window_stdev(gradient, 3) * 500
    # print(f'Gradient shape: {gradient.shape}')
    
    # std = np.array(std, dtype='float64')
    # print(f'Std type: {type(std)}')
    
    # threshold filter
    # threshold = filters.threshold_sauvola(std)
    # binary = (std > threshold*1)
    
    # snake
    # # Localising the circle's center at 220, 110
    # x1 = 200 + 100*np.cos(np.linspace(0, 2*np.pi, 500))
    # x2 = 100 + 100*np.sin(np.linspace(0, 2*np.pi, 500))
     
    # # Generating a circle based on x1, x2
    # snake = np.array([x1, x2]).T
     
    # # Computing the Active Contour for the given image
    # np_snake = active_contour(std, snake)
    
    # felzensz = felzenszwalb(gradient, scale = 2, sigma=5, min_size=1000)
    
    # # # plotting
    # fig, ax = plt.subplots(2,2,figsize=(20,20))
    # ax[0, 0].imshow(relative, cmap='gray')
    # ax[0, 0].set_title('Original')
    # ax[0, 1].imshow(gradient, cmap='cool')
    # ax[0, 1].set_title('Gradient')
    # ax[1, 0].imshow(std, cmap='cool')
    # ax[1, 0].set_title('StDev')
    # ax[1, 1].imshow(felzensz)
    # ax[1, 1].set_title('Felzensz')
    # # ax[1, 0].plot(np_snake[:, 0], np_snake[:, 1],'-b', lw=5)
    # # ax[1, 1].imshow(binary)
    # # ax[1, 1].set_title('Binary')
    # plt.show()
    
    # # update spatial information
    # out_meta.update({"driver": "GTiff",
    #                   "height": felzensz.shape[0],
    #                   "width": felzensz.shape[1],
    #                   "transform": out_transform})
    
    # # output raster name
    # outraster_name = os.path.join(os.path.dirname(band), 'felzensz.tif')
    
    # masked = np.where(felzensz == 1, felzensz, relative)
    # masked = np.where(felzensz == 1, nodata, relative)
    
    # with rio.open(outraster_name, "w", **out_meta) as dest:
    #     dest.write(masked, 1) # write array to raster
        
    # print(f'Output: {outraster_name}')
    
    # image = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 0, 0, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    #                   [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], np.uint8)
    # # plt.imshow(image);
    
    # imfilt = correlate(magnitude, image)
    # plt.imshow(imfilt)
    
    # print(type(imfilt))
    
    # segment = slic(std, n_segments=10, compactness=10)
    # plt.imshow(segment)
    
    # code that might be useful later
    
    # fig = plt.figure()
    # plt.gray()  # show the filtered result in grayscale
    
    # input_ = np.fft.fft2(relative)
    # result = ndimage.fourier_ellipsoid(input_, size=3)
    # result = ndimage.median_filter(relative, size=3)
    
    # fshift = np.fft.fftshift(input_)
    # fhp = fshift
    
    # rows, cols = relative.shape
    # crow,ccol = int(rows/2), int(cols/2)
    
    # remove the low frequencies by masking with a rectangular window of size 60x60
    # High Pass Filter (HPF)
    # fhp[crow-5:crow+5, ccol-5:ccol+5] = 0
    # f_ishift = np.fft.ifftshift(fhp)
    
    # # inverse fft to get the image back 
    # img_back = np.fft.ifft2(fshift)
    # result = np.abs(img_back)
    # result = result.real
    
    # result = np.fft.ifft2(result)
    
    # ax1 = fig.add_subplot(121)  # left 
    # ax2 = fig.add_subplot(122)  # middle
    # ax3 = fig.add_subplot(123) # right
    # result = ndimage.median_filter(relative, size=3)
    # mask = result == -9999
    # result = np.ma.masked_array(result, mask)
    # ax1.imshow(relative)
    # ax2.imshow(fshift.real)
    # ax3.imshow(result)
    # plt.show()
    
    # mask numpy array
    # mask = relative == -9999
    # relative = np.ma.masked_array(relative, mask)
    
    # numpy shapes
    # print(np.shape(relative))
    # print(np.min(relative), np.max(relative))
    # print(relative)
    
    # imshow(relative, plugin=) # https://scikit-image.org/docs/dev/api/skimage.io.html
    
    # rasterio object options
    # print(dark_image.name)
    # print(dark_image.mode)
    # print(dark_image.closed)
    # print(dark_image.count)
    # print(dark_image.width)
    # print(dark_image.height)
    # print(dark_image.bounds)
    # print(dark_image.transform * (dark_image.width, dark_image.height))
    # print(dark_image.crs)
    
    # if __name__ == '__main__':
        