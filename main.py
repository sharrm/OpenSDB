#################################################
# Geog 462/562 Group 6 Final Project
# Absolute Bathymetry from Satellite Derrived Bathymetry (relative) and Satellite Lidar using ICESat-2 and
# Sentinal/Landsat imagery

# Identify user defined function files
import SDB_Functions as sdb
import linear_regression as slr

#######################################################
# Step 1: Mask the area of interest for the imagery data

# Identify the input files
# maskSHP = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\Final Project\Other\clipper.shp" # in_shp
# blueInput = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\NewFinal\Sentinel2\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_492.tif" # Sentinel-2 band 
# greenInput = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\NewFinal\Sentinel2\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_560.tif" # Sentinel-2 band
# redInput = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\NewFinal\Sentinel2\S2A_MSI_2021_12_01_16_05_11_T17RNH_rhos_665.tif" # Sentinel-2 band

maskSHP = r"P:\SDB\Anegada\anagada_mask.shp" # in_shp
blueInput = r"P:\SDB\Anegada\LC08_L2SP_004047_20200223_20200822_02_T1_SR_B2.TIF" # Sentinel-2 band
greenInput = r"P:\SDB\Anegada\LC08_L2SP_004047_20200223_20200822_02_T1_SR_B3.TIF" # Sentinel-2 band
redInput = r"P:\SDB\Anegada\LC08_L2SP_004047_20200223_20200822_02_T1_SR_B4.TIF" # Sentinel-2 band

# returns of list of masked bands for each wavelength
maskOutput = sdb.mask_imagery(redInput, greenInput, blueInput, maskSHP)

# display masked file output location for each wavelength
if maskOutput[0]:
    maskFilesList = maskOutput[1]
    print(f"The masked files are:\n"
          f"{maskFilesList[0]}\n"
          f"{maskFilesList[1]}\n"
          f"{maskFilesList[2]}\n")

    maskedBlue = maskFilesList[0]
    maskedGreen = maskFilesList[1]
    maskedRed = maskFilesList[2]
else:
    print("No masked files were returned from the file masking function.")


############################
# Step 2 Ratio of logs between bands / relative bathymetry

# Start with Green SDB (deeper)
TF, outraster_name, blue_image, ratioBlueArrayOutput, lnBlueArrayOutput, ratioImage = sdb.pSDBgreen(maskedBlue, 
                                                                                         maskedGreen, 
                                                                                         maskSHP, 
                                                                                         rol_name='anegada_ratioLogs01')

# returns boolean and the pSDB location
# if green_SDB_output[0
if TF:
    # greenSDB = green_SDB_output[1]
    greenSDB = outraster_name
else:
    print("No green SDB raster dataset was returned from the pSDBgreen function.")

# # can modify this to compute the ratio of logs between other bands
# # for shallow water, the blue and red have been utilized in the literature
# red_SDB_output = sdb.pSDBgreen (maskedBlue, maskedRed)
#
# if red_SDB_output[0]:
#     redSDB = red_SDB_output[1]
# else:
#     print("No green SDB raster dataset was returned from the pSDBgreen function.")


##############################
# Step 3 Simple linear regression

# Identify the ICESat-2 reference dataset
# icesat2 = r"G:\My Drive\OSU Work\Geog 462 GIS III Analysis and Programing\Final Project\ICESat2\icesat2_clipped.csv"
icesat2 = r"P:\SDB\Anegada\processed_ATL03_20200811115251_07200801_005_01_o_o_clipped.csv"

# Starting with the Green band:
# Identify other parameters
SDBraster = greenSDB
col = "green"
loc = "Anegada"

# Run the function to see the relationship between lidar depth and relative bathymetric depths
# (returns b0 and b1 as a tuple)
greenSLRcoefs = slr.slr(SDBraster, icesat2, col)

# # Red band next:
# # Identify other parameters
# SDBraster = redSDB
# col = "green"
# loc = "Key_Largo_Florida"
#
# # Run the function to see the relationship between lidar depth and relative bathymetric depths
# # (returns b0 and b1 as a tuple)
# greenSLR = slr(SDBraster, icesat2, col)

# # Next the Red Band:
# # Identify other parameters
# SDBraster = redSDB
# col = "red"
# loc = "Key_Largo_Florida"


################################
# Step 4 Apply SLR to relative bath

# Only green functionality is currently modeled
SDBraster = greenSDB
col = 'green'
loc = "Anegada"

tf, final_raster, true_bath = slr.bathy_from_slr(SDBraster, greenSLRcoefs, col, loc)

if final_raster[0]:
    print(f"The final raster is located at: {final_raster}")
else:
    print("Something went wrong with creating the lidar-based SDB raster.")