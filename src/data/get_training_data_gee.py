import pandas as pd
import ee
ee.Initialize()

def get_array_image():
    kernel_size = 256
    one_d_list = ee.List.repeat(1, kernel_size)
    list_of_one_d_lists = ee.List.repeat(one_d_list, kernel_size)
    kernel = ee.Kernel.fixed(kernel_size, kernel_size, list_of_one_d_lists)
    return _get_image().neighborhoodToArray(kernel)

def _get_image():
    return ee.Image([_get_l8(), _get_nlcd(), _get_hm()]).float()

def _get_l8():
    # Taken from https://github.com/google/earthengine-api/blob/master/python/examples/ipynb/UNET_regression_demo.ipynb, will have to take a closer look at some point. Also see https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1368_L8_C1-LandSurfaceReflectanceCode-LASRC_ProductGuide-v3.pdf
    def mask_l8(image):
      opticalBands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
      thermalBands = ['B10', 'B11']

      cloudShadowBitMask = ee.Number(2).pow(3).int()
      cloudsBitMask = ee.Number(2).pow(5).int()
      qa = image.select('pixel_qa')
      mask1 = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(
        qa.bitwiseAnd(cloudsBitMask).eq(0))
      mask2 = image.mask().reduce('min')
      mask3 = image.select(opticalBands).gt(0).And(
              image.select(opticalBands).lt(10000)).reduce('min')
      mask = mask1.And(mask2).And(mask3)
      return image.select(opticalBands).divide(10000).addBands(
              image.select(thermalBands).divide(10).clamp(273.15, 373.15)
                .subtract(273.15).divide(100)).updateMask(mask)

    return (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
            .filterDate('2016-01-01', '2016-12-31')
            .map(mask_l8)
            .median())

def _get_nlcd():
    nlcd_2016 = ee.Image('USGS/NLCD/NLCD2016')
    return nlcd_2016.select('impervious').rename('imp')

def _get_hm():
    return ee.Image('projects/GEE_CSP/HM/HM_ee_2017_v014_500_30').select('Hall').rename('hm')

def export_training_data(points, number_of_points, array_image, bands_to_export): 
    points_per_file = 2000
    number_of_files = _get_number_of_files(number_of_points, points_per_file)

    for file_i in range(number_of_files):
        _export_single_file(points, file_i, points_per_file)

def _get_number_of_files(number_of_points, points_per_file):
    number_of_full_files = number_of_points // points_per_file
    number_of_partial_files = 1 if number_of_points % points_per_file > 0 else 0
    return number_of_full_files + number_of_partial_files

def _export_single_file(points, file_i, points_per_file):
    number_of_shards = _get_number_of_shards(points_per_file)

    fcs = []
    for shard_i in range(number_of_shards):
        these_points = _get_points_for_shard(points, points_per_file, file_i, shard_i)
        these_points_with_values = feature_array_image.reduceRegions(
                collection=these_points,
                reducer=ee.Reducer.first(),
                scale=30)
        fcs.append(these_points_with_values)

    fc = ee.FeatureCollection(fcs).flatten()
    _export_to_gcs(fc, file_i)

def _get_number_of_shards(points_per_file, shard_size=10):
    number_of_full_shards = points_per_file // shard_size
    number_of_part_shards = 1 if points_per_file % shard_size > 0 else 0
    return number_of_full_shards + number_of_part_shards

def _get_points_for_shard(points, points_per_file, file_i, shard_i):
    these_ids = _get_ids_for_shard(file_i, points_per_file, shard_i)
    pt_filter = ee.Filter.inList('ID', these_ids)
    return points.filter(pt_filter)

def _get_ids_for_shard(file_i, points_per_file, shard_i, shard_size=10):
    file_start_i = file_i * points_per_file
    this_start = file_start_i + shard_i * shard_size
    this_end = min(file_start_i + number_of_points, (this_start + shard_size))
    return list(range(this_start, this_end))

def _export_to_gcs(fc, file_i):
    task = ee.batch.Export.table.toCloudStorage(
             collection=fc,
             bucket='lumonitor',
             fileNamePrefix='sample_data_' + str(file_i),
             fileFormat='TFRecord',
             selectors=bands
           )
    task.start()

if __name__ == '__main__':
    import sys
    input_csv = sys.argv[1]
    points = ee.FeatureCollection('users/jesse/lumonitor/training_points')
    number_of_points = points.size().getInfo()
    feature_array_image = get_array_image()
    bands = ['B1','B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11', 'hm', 'imp']
    export_training_data(points, number_of_points, feature_array_image, bands)
