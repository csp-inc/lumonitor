library(dplyr)
library(exactextractr)
library(raster)
library(sf)
library(tidyr)
library(tigris)
library(usdarnass)

output_file <- commandArgs(TRUE)[1]
irrigated_areas <- raster('data/irrigated_areas.tif')
crs <- projection(irrigated_areas)

state_sf <- states(cb=TRUE, resolution="5m", year=2018) %>%
  st_transform(crs)

state_volume <- nass_data(
  agg_level_desc="STATE",
  year="2018",
  short_desc="AG LAND, IRRIGATED - WATER APPLIED, MEASURED IN ACRE FEET",
  domain_desc="TOTAL") %>%
mutate(acre_feet_applied = as.numeric(gsub(',', '', Value))) %>%
dplyr::select("STATEFP"=state_fips_code, acre_feet_applied)

ac_per_sqm <- 1 / 4046.86

state_volume %>%
right_join(state_sf) %>%
dplyr::select(acre_feet_applied, geometry) %>%
st_as_sf %>%
mutate(
  state_irrigated_area_pixels=exact_extract(irrigated_areas, ., 'sum'),
  acre_feet_per_pixel=acre_feet_applied/ state_irrigated_area_pixels
  ) %>%
write_sf(output_file)
