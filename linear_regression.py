import rasterio
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import os


# if __name__ == "__main__":

def slr(in_raster, icesat2, color):
    # Load in our CSV
    pts = pd.read_csv(icesat2)

    # create a list of coordinate pairs from easting/northings in the csv
    # these need to be in (x,y) for the rasterio sample function
    coords = [(x, y) for x, y in zip(pts.E, pts.N)]

    src = rasterio.open(in_raster) # open the provided raster

    # # could potentially be another way to sample raster values:
    # sampled = rasterio.sample.sample_gen(in_raster, coords)

    # Add a column of raster values for the sampled locations using input csv E/N
    pts['Raster Value'] = [x[0] for x in src.sample(coords)]

    X = pts.drop(columns=['E', 'N', 'Z']) # keep only the raster values
    y = pts.Z # the reference Z values to use in the regression

    # simple linear regression
    reg = LinearRegression().fit(X, y)
    r2 = np.around(reg.score(X, y), decimals=3)
    b1 = np.around(reg.coef_, decimals=3)
    b0 = np.around(reg.intercept_, decimals=3)
    print(f"The r-squared value is: {r2}")
    print(f"The m0 value is: {b0}")
    print(f"The m1 value is: {b1}")

    pts['y_hat'] = reg.predict(X)

    # with pd.ExcelWriter(r'P:\SDB\Anegada\pandas.xlsx') as writer:
    #     pts.to_excel(writer)
        
    print(pts)    

    # plt.scatter(pts.Z, pts.y_hat, alpha=0.5)
    ax = plt.gca()
    # plt.plot(X, pts.y_hat, '.')
    plt.plot(X, y, '+', label='Data')
    # x_vals = np.array([np.min(X), np.max(X)])
    x_vals = np.array(ax.get_xlim())
    y_vals = b0 + b1 * x_vals
    plt.plot(x_vals, y_vals, '--', label='SLR Fit')
    # plt.text(np.min(x_vals), np.min(y), f'Y = {b0} + {b1}X\nR^2 = {r2}', fontsize=10)
    
    # plot formatting
    plt.text(np.min(x_vals), np.min(y), f'$Depth$ = ${{{b1}}}X$ + ${{{b0}}}$\n$R^2$ = ${{{r2}}}$', fontsize=10)
    plt.title('ICESat-2 vs pSDB')
    plt.xlabel(f'{color} pSDB (log)')
    plt.ylabel('ICESat-2 (m)')
    plt.legend()
    plt.grid()
    plt.show()

    # Create a tuple for return
    coefs = (b0, b1)
    return coefs

# read and write a new raster file using the coefficients from SLR
def bathy_from_slr(in_raster, coefsB0B1, color, location):
    # identify the coefficients
    b0 = coefsB0B1[0] # intercept
    b1 = coefsB0B1[1] # slope

    # # read in the raster
    # src = rasterio.open(in_raster)
    # src = src.read(1)
    # # Extract metadata
    # out_meta = src.meta

    # Capture the metadata and create an array from the raster file
    with rasterio.open(in_raster) as src:
        out_meta = src.meta
        nodata = out_meta['nodata']
        rastArray = src.read(1)

    # Based on the input color, conduct a raster math operation on the array to apply slr coefficients to relative
    # bathymetry. Here, we are multiplying by (-1) so down is positive
    if color == "green":
            true_bath = np.where(rastArray != nodata, -1*(b1 * rastArray + b0), rastArray)
    elif color == "red": # Reserved for future use
            true_bath = np.where(rastArray != nodata, -1*(b1 * rastArray + b0), rastArray)
    elif color == "intermediate": # Reserved for future use
            print(f"{color} is not yet functional.")
            return False, None
        
    # # for Python 3.10 or later with match switch functionality
    # match color:
    #     case "green":
    #         true_bath = -1*(b1 * rastArray + b0)
    #     case "red": # Reserved for future use
    #         true_bath = -1*(b1 * rastArray + b0)
    #     case "intermediate": # Reserved for future use
    #         print(f"{color} is not yet functional.")
    #         return False, None

    path = os.path.dirname(in_raster)
    outraster_name = os.path.join(path, f"SLR_bathymetry{color}.tif")
    with rasterio.open(outraster_name, "w", **out_meta) as dest:
        dest.write(true_bath, 1)

    return True, outraster_name, true_bath
