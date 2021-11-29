library(dplyr)
library(exactextractr)
library(raster)
library(sf)

input_st <- stack(list(
  "ag_2013" = "data/predictions/hag_s_2013_hm-2016_1637276166_2d4f1735/hag_s_2013_prediction_conus.tif",
  "ag_2016" = "data/predictions/hag_s_2016_hm-2016_1637276166_2d4f1735/hag_s_2016_prediction_conus.tif",
  "ag_2020" = "data/predictions/hag_s_2020_hm-2016_1637276166_2d4f1735/hag_s_2020_prediction_conus.tif",
  "trans_2013" = "data/predictions/htrans_2013_hm-2016_1636478645_6b6a8ce5/htrans_2013_prediction_conus.tif",
  "trans_2016" = "data/predictions/trans_2016_max.tif",
  "trans_2020" = "data/predictions/trans_2020_max.tif",
  "urban_2013" = "data/predictions/hurban_c_2013_hm-2016_1637623568_31c2a6cd/hurban_c_2013_prediction_conus.tif",
  "urban_2016" = "data/predictions/urban_2016_max.tif",
  "urban_2020" = "data/predictions/urban_2020_max.tif"
))

states <- read_sf("/vsizip/data/azml/cb_2019_us_state_5m.zip") %>%
  dplyr::filter(!STUSPS %in% c("AK", "GU", "PR", "VI", "MP", "AS", "HI")) %>%
  st_transform(crs(input_st))
states$area_km2 <- units::set_units(st_area(states), "km^2")

stats <- exact_extract(input_st, states, fun = "mean", append_cols = "STATEFP") %>%
  right_join(states) %>%
  mutate(stats, across(starts_with('mean'), list(area_km2=function(x) x * area_km), .names='{substr({.col},6,nchar({.col}))}_{.fn}') %>%
  st_as_sf() %>%
  st_transform("EPSG:4326")

# conver mean.ag_2013 to ag_2013_mean
names(stats) = sub('^mean.(.*)$', '\\1_mean', names(stats))

write_sf(stats, "data/summary_stats.geojson")
