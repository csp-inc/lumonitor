import ee
ee.Initialize()

solar = ee.FeatureCollection('users/jesse/lumonitor/solar_points')
aoi = ee.FeatureCollection('users/jesse/aft-app/conus')

points = ee.FeatureCollection.randomPoints(aoi, solar.size(), seed=1337)

task = ee.batch.Export.table.toAsset(collection=points, 
        assetId='users/jesse/lumonitor/non_solar_points')

task.start()
