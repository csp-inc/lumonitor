import ee
import solar_constants as sc

ee.Initialize()


def add_label(geom, label):
    def really_add_label(feature):
        return feature.set('solar', label)

    return geom.map(really_add_label)

solar = add_label(ee.FeatureCollection('users/jesse/lumonitor/solar_points'), 1)

random = add_label(ee.FeatureCollection('users/jesse/lumonitor/non_solar_points'), 0)

def maskL8sr(image):
    cloudShadowBitMask = ee.Number(2).pow(3).int()
    cloudsBitMask = ee.Number(2).pow(5).int()
    qa = image.select('pixel_qa')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0) \
             .And(qa.bitwiseAnd(cloudsBitMask).eq(0))
    return image.updateMask(mask).select(sc.BANDS).divide(10000)


# The image input data is a 2018 cloud-masked median composite.
image = sc.L8SR.filterDate('2018-01-01', '2018-12-31').map(maskL8sr).median()
all_data = solar.merge(random)

sample = image.sampleRegions(collection=all_data, properties=['solar'], scale=30) \
              .randomColumn()

training = sample.filter(ee.Filter.lt('random', 0.7))
test = sample.filter(ee.Filter.gte('random', 0.7))

task = ee.batch.Export.table.toCloudStorage(
         collection=training,
         fileNamePrefix='solar_training',
         bucket='aft-saf',
         fileFormat='TFRecord',
         selectors=sc.FEATURE_NAMES)

task.start()

task = ee.batch.Export.table.toCloudStorage(
         collection=test,
         fileNamePrefix='solar_test',
         bucket='aft-saf',
         fileFormat='TFRecord',
         selectors=sc.FEATURE_NAMES)

task.start()
