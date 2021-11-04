library(dplyr)
library(exactextractr)
library(raster)
library(sf)

input_st <- stack(list(
  "2013" = "data/conus_2013_prediction.tif",
  "2016" = "data/conus_2016_prediction_max.tif",
  "2020" = "data/conus_2020_prediction_max.tif"
))

states <- read_sf("/vsizip/data/azml/cb_2019_us_state_5m.zip") %>%
  dplyr::filter(!STUSPS %in% c("AK", "GU", "PR", "VI", "MP", "AS", "HI")) %>%
  st_transform(crs(input_st))

stats <- exact_extract(input_st, states, fun = "mean", append_cols = "STATEFP") %>%
  right_join(states) %>%
  rename(mean_2013 = mean.X2013, mean_2016 = mean.X2016, mean_2020 = mean.X2020) %>%
  st_as_sf() %>%
  st_transform("EPSG:4326")

browser()
write_sf(stats, "data/summary_stats.geojson")
