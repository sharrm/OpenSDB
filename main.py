# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

# personel functions
from clustering import Cluster
from filtering import Filters
from fourier import Fourier
from raster_functions import ReadWriteFunctions, ArrayOperations
import misc
from segmentation import Segmentation

# open source
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import numpy as np
import os

# check if not using Covelli version as interpreter
if not 'Covelli' in sys.executable:
    print(f'Using Python interpreter: {sys.executable}')
else:
    print(f'\n* Using interpreter: {sys.executable}\n')

if __name__ == '__main__':
    
    ## image file input
    band = r"C:\Users\matthew.sharr\Documents\Work\Sat-Bathy\Test Files\masked.tif"
    # band = r"C:\Users\matthew.sharr\Pictures\Best-Black-and-white-pictures.jpg"
    
    ## array from input
    if os.path.isfile(band):
        bandarray = ReadWriteFunctions().image_array(band)
        metadata = ReadWriteFunctions().meta_data(band)
        nodata = metadata['nodata']
    
    
    ## fourier transform functions
    frequency = Fourier().freq(bandarray)
    frequencyshifted = Fourier().freqshifted(frequency)
    magnitude = Fourier().magnitude_spectrum(bandarray)
    freq_pack = Fourier().fftpack(bandarray)


    ## frequency filters
    # hpf = Filters().hpf(bandarray, frequency, 50)
    fourier_ellipse = Filters().fourier_ellipse(frequency, 2)
    fourier_gaussian = Filters().fourier_gaussian(frequency, 2)
    # fourier_gaussian = Fourier().freqshifted(fourier_gaussian)
    mag_fg = 20 * np.log(abs(fourier_gaussian))
    custom_filt = Filters().custom(freq_pack)
    
    
    ## spatial filters
    # spatial = Filters().median_filter(bandarray, size=3)
    
    
    ## inverse fourier transform functions
    # hpf_ifft = Fourier().inverse(hpf)
    fourier_ellipse = Fourier().inverse(fourier_ellipse)
    fourier_gaussian = Fourier().inverse(fourier_gaussian)
    inverse = Fourier().inverse_shifted(frequencyshifted)
    inv_pack = Fourier().ifftpack(custom_filt)
    
    
    ## spatial calculations
    gradient = ArrayOperations().gradient(bandarray)
    
    
    ## segmentation
    felensz = Segmentation().felensz(gradient, scale = 2, sigma=5, min_size=1000)
    
    
    ## masking
    masked = ArrayOperations().mask(bandarray, felensz, nodata)
    masked[masked == -9999] = np.nan
        
    
    ## plotting
    fig, ax = plt.subplots(3,2,figsize=(25,25))
    ax[0, 0].imshow(bandarray, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(magnitude.real, cmap='gray')
    ax[0, 1].set_title('Magnitude')
    ax[1, 0].imshow(mag_fg.real, cmap='gray')
    ax[1, 0].set_title('Fourier Gaussian Filter')
    ax[1, 1].imshow(fourier_gaussian.real, cmap='gray')
    ax[1, 1].set_title('After Fourier Gaussian Filter')
    ax[2, 0].imshow(fourier_ellipse.real, cmap='gray')
    ax[2, 0].set_title('After Fourier Ellipse Filter')
    ax[2, 1].imshow(masked, cmap='gray')
    ax[2, 1].set_title('Masked with Felensz')
    # ax[2, 1].imshow(plt.imshow(np.abs(inv_pack.real), cmap='gray'))
    # ax[2, 1].set_title('FFT Pack Segmentation')
    plt.show()

    
    band = None
    
    


