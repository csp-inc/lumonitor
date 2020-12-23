from pyrasterframes.utils import create_rf_spark_session
from pyrasterframes.rasterfunctions import st_polygonFromText, st_intersects, st_geometry
from pyspark.sql.functions import lit

spark = create_rf_spark_session()

conus = spark.read.geojson('data/aoi_conus.geojson')

# For now, just use aws catalog
l8 = spark.read.format('aws-pds-l8-catalog').load()

l8 = l8 \
        .withColumn('geom', st_geometry(l8.bounds_wgs84)) \
        .withColumn('conus_text', lit(conus.first()['geometry'].wkt)) \
        .withColumn('conus', st_polygonFromText('conus_text'))


l8.filter(st_intersects(l8.geom, l8.conus)) \
  .filter(l8.acquisition_date > '2016-01-01') \
  .filter(l8.acquisition_date < '2016-12-31')


# Masking code below
b1 = 'work/HLS.L30.T10TFT.2016001.v1.4_01.tif'
b11 = 'work/HLS.L30.T10TFT.2016001.v1.4_11.tif'
l8 = spark.read.raster([[b1, b11]]).withColumnRenamed('proj_raster_0', 'b1').withColumnRenamed('proj_raster_1', 'b11')
l8.printSchema()
from pyrasterframes.rasterfunctions import rf_tile

# Display data
#tile = b1.select(rf_tile('proj_raster'))
# tile


mask_ct = l8.select(rf_cell_type('b11')).distinct()
mask_ct
# uint8raw, neeed to change types to set nodata

mask_ct = CellType('uint8').with_no_data_value(-1000)
mask_ct

cirrus = 0b1
cloud = 0b10
adjacent_cloud = 0b100
cloud_shadow = 0b1000
high_aerosol = 0b11000000
med_aerosol = 0b10000000

clouds = [cirrus, cloud, adjacent_cloud, cloud_shadow, high_aerosol, med_aerosol]


l8 = l8.withColumn('b11', rf_convert_cell_type('b11', mask_ct)) \
       .withColumn('b11', rf_mask_by_values('b11', 'b11', clouds)) \
       .withColumn('b1', rf_mask_by_values('b1', 'b11', clouds))

# l8.select(rf_tile('b1'))

