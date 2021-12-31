library(dplyr)
library(exactextractr)
library(raster)
library(sf)
library(tidyverse)

args <- commandArgs(trailingOnly = TRUE)
input_st <- stack(list(
  "ag_2013" = args[1],
  "ag_2016" = args[2],
  "ag_2020" = args[3],
  "trans_2013" = args[4],
  "trans_2016" = args[5],
  "trans_2020" = args[6],
  "urban_2013" = args[7],
  "urban_2016" = args[8],
  "urban_2020" = args[9]
))
output_geojson <- args[10]

states <- read_sf("/vsizip/data/azml/cb_2019_us_state_5m.zip") %>%
  dplyr::filter(!STUSPS %in% c("AK", "GU", "PR", "VI", "MP", "AS", "HI")) %>%
  st_transform(crs(input_st))
states$area_km2 <- units::set_units(st_area(states), "km^2")

stats <- exact_extract(input_st, states, fun = "mean", append_cols = "STATEFP")

stats <- stats %>%
  right_join(states) %>%
  mutate(across(starts_with("mean"), list(area_km2 = function(x) {
    (x / 100) * area_km2
  }), .names = "{substr({.col},6,nchar({.col}))}_{.fn}")) %>%
  st_as_sf() %>%
  st_transform("EPSG:4326")

# conver mean.ag_2013 to ag_2013_mean
names(stats) <- sub("^mean.(.*)$", "\\1_mean", names(stats))

if (file.exists(output_geojson)) unlink(output_geojson)
write_sf(stats, output_geojson)
