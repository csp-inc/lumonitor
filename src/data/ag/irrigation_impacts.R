library(magrittr)
library(terra)

irrigated_areas <- rast("data/irrigated_areas.tif")
terraOptions(datatype = "FLT2S", memfrac = 0.9)

for (year in c("2016", "2020")) {
  daymet <- rast(paste0("data/daymet_precip_", year, ".tif"))
  ssebop <- rast(paste0("data/ssebop_", year, ".tif"))

  net_water <- resample(daymet, ssebop) - ssebop
  mm_per_feet <- 304.8
  sqm_per_acre <- 4046.86
  net_acre_feet <- net_water / mm_per_feet * abs(prod(res(irrigated_areas))) / sqm_per_acre

  browser()
  resample(net_acre_feet, irrigated_areas,
    filename = paste0("data/net_acre_feet_", year, ".tif"),
    overwrite = TRUE, gdal = c("COMPRESS=LZW", "PREDICTOR=3")
  )
}
