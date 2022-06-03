import numpy as np
from scipy import fftpack, integrate
# import pyfits
import pylab as py
import rasterio
import glob
import os
# from matplotlib.pyplot import plt
# import radialProfile

def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    image - The 2D image
    center - The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
    
    """
    # Calculate the indices from the image
    y, x = np.indices(image.shape)

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=float)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr

    return radial_prof

def PSD(img):    # rasters
    # image = pyfits.getdata(‘myimage.fits’)
    
    # for band in rasters:
              
    # Take the fourier transform of the image.
    # img = rasterio.open(band)
    # img = img.read(1)
    
    # img = np.pad(img, ((1,1),(1,1)), 'constant')
    
    # img[img == -9999] = 0
    
    F1 = fftpack.fft2(img)
    
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift( F1 )
    
    # Calculate a 2D power spectrum
    psd2D = np.abs( F2 )**2
    
    # Calculate the azimuthally averaged 1D power spectrum
    psd1D = azimuthalAverage(psd2D)
    
    # Now plot up both
    # py.figure(1)
    # py.clf()
    # py.imshow( np.log10( img ), cmap=py.cm.Greys)
    
    # py.figure(2)
    # py.clf()
    # py.imshow( np.log10( psd2D ))
    
    # bandname = os.path.basename(band)[:-4]
    # py.title(bandname)
    # py.legend(loc='upper right')
    
    # threshold = np.median(psd2D)
    # py.imshow(psd2D, cmap=py.cm.hot, interpolation='none', vmax=threshold)
    
    # yrange = psd1D[50:]
    # xrange = np.arange(50, 50 + len(yrange), 1)
    yrange = psd1D[:]
    xrange = np.arange(0, len(yrange), 1)

    trapz = np.trapz(yrange, xrange)
    # print(f'Energy {band}: {trapz:.3f}')
        
        # py.semilogy(psd1D, label = bandname)
        # # py.semilogy(yrange, label = bandname)
        # py.xlabel('Spatial Frequency')
        # py.ylabel('Power Spectrum')
        # py.title('Radially Averaged PSD')
        
    # handles, labels = py.gca().get_legend_handles_labels()
    # # order = [2, 0, 3, 1, 5, 4]
    # order = [3, 2, 1, 0]
    
    # h = [handles[i] for i in order]
    # l = [labels[i] for i in order]
    
    # py.legend(h, l, loc='upper right')
    # py.savefig(os.path.join(r'U:\ce567\sharrm\Final\Figures', 'SamplesRAPSD.png'), dpi=300)
    # py.grid()
    # py.show()

    # py.clf()
        
    return trapz

# rasters = []

# for tif in glob.glob(r'P:\SDB\Test_Files\Samples\Rasters\*.tif', recursive=False):
#     rasters.append(tif)

# rasters.sort()

# psd1d, psd2d, img = PSD(rasters)

## integration 
# X = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]
# rA = -1 / np.array([-.0053, -.0052, -.0050, -.0045, -.0040, -.0033, -.0025, -.0018, -.00125, -.0010])
# py.plot(X, rA)
# py.show()
# Int = np.trapz(rA, X)
# print(f'Trapz: {Int}')

# this works same as in matlab
# test = psd1d[150:]
# arange = np.arange(150, 150 + len(test), 1)
# print(len(test))

# trapz = np.trapz(test, arange)
# py.plot(arange, test)
# py.show()

# print(test)
# print(arange)

# print(f'Trapz: {trapz}')




# threshold = np.mean(psd2d)

# fig = py.figure(figsize=(6, 4))

# ax = fig.add_subplot(111)
# ax.set_title('colorMap')
# im = py.imshow(psd2d, cmap=py.cm.hot, interpolation='none', vmax=threshold)

# cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
# cax.get_xaxis().set_visible(False)
# cax.get_yaxis().set_visible(False)
# cax.patch.set_alpha(0)
# cax.set_frame_on(False)
# cbar = fig.colorbar(im, extend='max')
# cbar.cmap.set_over('green')
# py.show()

    
    
    
    