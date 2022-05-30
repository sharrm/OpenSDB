import numpy as np
from scipy import fftpack
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

def PSD(rasters):   
    # image = pyfits.getdata(‘myimage.fits’)
    
    for band in rasters:
              
        # Take the fourier transform of the image.
        img = rasterio.open(band)
        img = img.read(1)
        
        img = np.pad(img, ((1,0),(0,1)), 'constant')
        
        img[img == -9999] = 0
        
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
        
        py.semilogy(psd1D, label = os.path.basename(band)[:-4])
        py.xlabel('Spatial Frequency')
        py.ylabel('Power Spectrum')
        py.title('Radially Averaged PSD')
        # py.title(os.path.basename(band)[:-4])
        py.legend(loc='upper right')
        
        # threshold = np.median(psd2D)
        # py.imshow(psd2D, cmap=py.cm.hot, interpolation='none', vmax=threshold)
        
    
    # handles = ['Noiseless', 'SomeNoise', 'AlmostHalf', 'MoreHalf', 'AlmostFull',  'Full']
    # order = [1, 2, 3, 4, 5, 6]
    
    
    # py.savefig(os.path.join(r'P:\SDB\Test_Files\Noise_Files\Figures', 'RAPSD.png'), dpi=300)
    
    py.show()
    py.clf()
        
    return psd1D, psd2D, img

rasters = []

for tif in glob.glob(r'P:\SDB\Test_Files\Noise_Files\*.tif', recursive=False):
    rasters.append(tif)

psd1d, psd2d, img = PSD(rasters)

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

    
    
    
    