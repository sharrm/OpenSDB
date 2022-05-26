# -*- coding: utf-8 -*-
"""
@author: sharrm
"""

# personel functions
from clustering import Cluster
from filtering import Filters
from fourier import Fourier
import raster_functions
import misc

# open source
import matplotlib.pyplot as plt
import sys

# check if not using Covelli version as interpreter
if not 'Covelli' in sys.executable:
    print(f'Using Python interpreter: {sys.executable}')
else:
    print(f'\n* Using interpreter: {sys.executable}\n')

if __name__ == '__main__':
    
    # image file input  
    band = r"P:\SDB\Florida Keys\Popcorn\Test_Files\masked.tif"
    
    # array from input
    bandarray = raster_functions.RasterFunctions().image_array(band)
    metadata = raster_functions.RasterFunctions().meta_data(band)
    
    # fft functions
    frequency = Fourier().forward(bandarray)
    magnitude = Fourier().magnitude_spectrum(bandarray)

    # frequency filters
    hpf = Filters().hpf(bandarray, frequency, 50)
    
    # ifft
    # hpf_ifft = Fourier().inverse(hpf)
    fourier_ellipse = Fourier().inverse(Filters().fourier_ellipse(frequency, 50))
    inverse = Fourier().inverse(bandarray)
    
    # spatial filters
    # spatial = Filters().median_filter(bandarray, size=3)
        
    # plotting
    fig, ax = plt.subplots(2,2,figsize=(20,20))
    ax[0, 0].imshow(bandarray, cmap='gray')
    ax[0, 0].set_title('Original')
    ax[0, 1].imshow(magnitude.real, cmap='gray')
    ax[0, 1].set_title('Magnitude')
    ax[1, 0].imshow(fourier_ellipse.real, cmap='gray')
    ax[1, 0].set_title('HPF Filter')
    ax[1, 1].imshow(inverse.real, cmap='gray')
    ax[1, 1].set_title('After FFT')
    plt.show()

    # kmeans clustering
    # kmeans = Cluster().kmeans(bandarray)
    
    


