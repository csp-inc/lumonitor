library(terra)
library(fasterize)

cdl_file <- 'data/2020_30m_cdls.zip'
if (!file.exists(cdl_file)) {
  cdl_url <- 'https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/2020_30m_cdls.zip'
  download.file(cdl_url, cdl_file)
}

terraOptions(datatype='INT1U')

cdl_2020 <- rast(file.path('/vsizip', cdl_file, '2020_30m_cdls.img'))
levels(cdl_2020) <- NULL

rcl_tbl <- matrix(c(0,61,65,78,194,61,65,78,194,256,1,0,1,0,1), ncol=3)
cropland_2020 <- classify(cdl_2020, rcl_tbl)

writeRaster(
  cropland_2020,
  'data/cropland_2020.tif',
  gdal=c(COMPRESS="LZW", PREDICTOR=2)
)
