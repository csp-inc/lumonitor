# Code taken from https://github.com/csp-inc/ag-connectivity/blob/main/scripts/Python/ag-resist-for-jesse.ipynb
import math

import ee
import numpy as np
import pandas as pd

import export_hm

# Initialize ee
ee.Initialize()

# Choose resistance value for water (btwn 1 and ~1000)
water_res = 900
# COLLECT DATA AND SET CONSTANTS

# Get states and full conus AOI boundaries
states = ee.FeatureCollection("TIGER/2018/States")
AOI = ee.FeatureCollection("projects/GEE_CSP/thirty-by-thirty/aoi_conus")

# Get FUT Landcover layer for 2016 (10m res) - Mostly for crs at this point
lcov = ee.Image("projects/GEE_CSP/AFT_FUT/LCU_2015_v6")
gCRS = lcov.projection().getInfo()["crs"]

# Set H values for ag land classes
crop = 0.5
pasture = 0.4
rangeland = 0.2
wood = 0.2

# Matrix of ones
ones = ee.Image(1)

# Get state-level crop planting dates
st_dates = pd.read_csv(r"/data/state-planting-dates.csv")
st_names = st_dates["Location"].tolist()

# Get Plant Hardiness Zones map
phz = ee.FeatureCollection(
    "projects/GEE_CSP/aft-connectivity/plant-hardiness-zones-2012"
)
phzVals = ee.List(phz.aggregate_array("ZONE_gr")).distinct().getInfo()
## DEFINE FUNCTIONS

# Function for calculating NDVI from surface relectance bands (B1 = red, B2 = NIR)
def addNDVI(image):
    ndvi = image.normalizedDifference(["sur_refl_b02", "sur_refl_b01"]).rename("NDVI")
    return image.addBands(ndvi)


# Function for extracting QA bit info on cloudy pixels from MOD09GA
def getQABits(image, start, end, newName):
    pattern = 0
    seq = range(start, end + 1)
    for i in seq:
        pattern += math.pow(2, i)  # pattern = pattern + Math...
    return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start)


# Function for masking cloudy pixels
def clear(image):
    img = image.select("state_1km")
    cloudBits = getQABits(img, 0, 1, "Clear").expression("b(0) == 0 || b(0) == 3")
    return image.updateMask(cloudBits)


# For 16-day ndvi, multiply by 0.0001 to bring back to original scale
def modisScale(image):
    return image.multiply(0.0001)


# Calculate centered values
def centerVI(image):
    m = image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=AOI.geometry(), maxPixels=1e11, scale=250
    ).get("NDVI_variance")
    c = image.subtract(ee.Image(ee.Number(m)))
    return c


# PROCESS AG DATA

# Mask everything but agricultural cover types and their low density residential counterparts
# 1 = Crop, 2 = Pasture, 3 = Range, 5 = Woodland in crop
# Recode all non-ag types as 0, recode all LDR ag types as standard ag
ag = lcov.expression(
    "(b(0) == 1101 | b(0) == 1) ? 1"
    + ": (b(0) == 1102 | b(0) == 2) ? 2"
    + ": (b(0) == 1103 | b(0) == 3) ? 3"
    + ": (b(0) == 1105 | b(0) == 5) ? 5"
    + ": 0"
).clip(AOI)

# Binary layers for ag/not ag
isAg = ag.gt(0)
notAg = ag.eq(0)
isCrop = ag.eq(1)
isPasture = ag.eq(2)
isRange = ag.eq(3)
isWood = ag.eq(5)

# Baseline H values for each ag type
# H values for crop (1) and pasture (2) based on Theobald 2013 and Brown and Vivas 2005
hagTemp = (
    ag.expression(
        "(b(0) == 1) ? crop"
        + ": (b(0) == 2) ? pasture"
        + ": (b(0) == 3) ? rangeland"
        + ": (b(0) == 5) ? wood"
        + ":(b(0))",
        {"crop": crop, "pasture": pasture, "rangeland": rangeland, "wood": wood},
    )
    .rename("Hag")
    .clamp(0, 1)
)

# CALCULATE cvNDVI BY STATE
# Using state-specific growing season start and end dates, get all relevant MODIS NDVI
# images for a state, calculate cvNDVI, and then mosaic all states

vi_list = []
for i in range(len(st_names)):
    st_ft = states.filter(ee.Filter.eq("NAME", st_names[i]))
    # Justin: d3 is 2016 and d4 is 2017
    #    d1 = st_dates[st_dates["Location"] == st_names[i]][["st1", "end1"]].values[0]
    #    d2 = st_dates[st_dates["Location"] == st_names[i]][["st2", "end2"]].values[0]
    #    d3 = st_dates[st_dates["Location"] == st_names[i]][["st3", "end3"]].values[0]
    #    d5 = st_dates[st_dates["Location"] == st_names[i]][["st5", "end5"]].values[0]
    #    dateFilter = ee.Filter.Or(
    #        ee.Filter.date(d1[0], d1[1]),
    #        ee.Filter.date(d2[0], d2[1]),
    #        ee.Filter.date(d3[0], d3[1]),
    #        ee.Filter.date(d4[0], d4[1]),
    #        ee.Filter.date(d5[0], d5[1]),
    #    )
    d4 = st_dates[st_dates["Location"] == st_names[i]][["st4", "end4"]].values[0]
    dateFilter = ee.Filter.date(d4[0], d4[1])

    viCol = ee.ImageCollection("MODIS/006/MOD13Q1").filter(dateFilter).select("NDVI")
    vi = viCol.map(modisScale)

    # Calculate variance and sd of non-masked pixels
    viVar = vi.reduce(ee.Reducer.variance()).clip(st_ft)
    viMean = vi.reduce(ee.Reducer.mean()).clip(st_ft)
    viSD = viVar.sqrt()
    viCV = viSD.divide(viMean)

    # Add cvNDVI image to list
    vi_list.append(viCV)

# Mosaic all states together
ndviCV = ee.ImageCollection(vi_list).mosaic()
# ADJUST AG H VALUE BY CENTERED cvNDVI FOR CROP AND PASTURE

# ----------------------------------------------
# GET MEAN CENTERED cvNDVI WITH MEAN FOR EACH PLANT HARDINESS ZONE
# Subset to crop and pasture
VIcrop = ndviCV.updateMask(isCrop)
VIpast = ndviCV.updateMask(isPasture)

cropList = []
pastList = []
for i in range(len(phzVals)):
    zone = phz.filter(ee.Filter.eq("ZONE_gr", phzVals[i]))  # Choose hardiness zone
    cZone = VIcrop.clip(zone)  # Clip crop to zone
    pZone = VIpast.clip(zone)  # Clip pasture to zone

    # Test whether any crop or pasture pixels occurr in this PHZ
    ctest = (
        cZone.reduceRegion(ee.Reducer.mean(), AOI.geometry(), 5000)
        .get("NDVI_variance")
        .getInfo()
    )
    ptest = (
        pZone.reduceRegion(ee.Reducer.mean(), AOI.geometry(), 5000)
        .get("NDVI_variance")
        .getInfo()
    )

    # If so, center and add to image list
    if ctest is not None:
        cZone_center = centerVI(cZone)  # Mean-center crop values
        cropList.append(cZone_center)  # Add to image list
    if ptest is not None:
        pZone_center = centerVI(pZone)  # Mean-center pasture values
        pastList.append(pZone_center)  # Add to image list

VIcrop_center = ee.ImageCollection(
    cropList
).mosaic()  # Mosaic phz-specific images back into full image
VIpast_center = ee.ImageCollection(
    pastList
).mosaic()  # Mosaic phz-specific images back into full image

# Bound values of above layer to make sure H values stay in 0-1 range.
# This affects << 1% of pixels in each layer
VIcrop_center = VIcrop_center.clamp(-0.4, 0.4)
VIpast_center = VIpast_center.clamp(-0.4, 0.4)

# ----------------------------------------------
# COMBINE WITH OTHER AG COVER TYPES AND ADD TO BASELINE H VALLUES
# Make placeholders for other ag types and non-ag
PHrangeland = ee.Image(0).updateMask(isRange).rename("NDVI_variance")
PHwood = ee.Image(0).updateMask(isWood).rename("NDVI_variance")
PHnotag = ee.Image(0).updateMask(notAg).rename("NDVI_variance")

# Collect all into image collection and collapse to single band
agVI = ee.ImageCollection.fromImages(
    [VIcrop_center, VIpast_center, PHrangeland, PHwood, PHnotag]
).mosaic()

# Apply NDVI adjustment to Hag layer
# Reset non-ag areas to zero using isAg
hag = hagTemp.add(agVI).multiply(isAg).clamp(0, 1)


def edge_effect(img: ee.Image) -> ee.Image:
    resolutionOut = 30
    # Set some constants
    distance = min(255 * resolutionOut, 10000)
    halfLife = 500
    HMeeMax = ee.Image(0)

    seq = np.arange(0.1, 1.05, 0.05)
    for i in seq:
        HMt = img.gt(i)
        # dt: filter out small "salt" patches
        # JA: keeping this at 90 for now
        HMt = HMt.focal_mean(90, "circle", "meters")
        HMt = HMt.gt(0.5)

        # dt: calculate Euc distance away from 1s
        HMtdistance = HMt.distance(ee.Kernel.euclidean(distance, "meters"), False)
        HMee = (
            ee.Image(i)
            .multiply(ee.Image(0.5).pow(HMtdistance.divide(ee.Image(halfLife))))
            .unmask(0)
        )
        HMeeMax = HMeeMax.max(HMee)

    return ones.subtract(ones.subtract(HMeeMax).multiply(ones.subtract(img)))


hag_ee = edge_effect(hag)

export_hm(hag_ee, "hag_suraci_2017")
