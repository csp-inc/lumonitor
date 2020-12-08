from pyrasterframes.utils import create_rf_spark_session
from pyrasterframes.rasterfunctions import st_polygonFromText, st_intersects
spark = create_rf_spark_session()

conus = spark.read.geojson('data/aoi_conus.geojson')

# For now, just use aws catalog
l8 = spark.read.format('aws-pds-l8-catalog').load() \
          .withColumn('geom', st_geometry(l8.bounds_wgs84)) \
          .withColumn('conus_text', lit(conus.first()['geometry'].wkt)) \
          .withColumn('conus', st_polygonFromText('conus_text')) \
          .filter(st_intersects(l8.geom, l8.conus)) \
          .filter(l8.acquisition_date > '2016-01-01') \
          .filter(l8.acquisition_date < '2016-12-31')
