import ee
ee.Initialize()

L8SR = ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
# Use these bands for prediction.
BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7']
FEATURE_NAMES = list(BANDS)
FEATURE_NAMES.append('solar')
N_CLASSES = 2
