library(dplyr)
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
select("STATEFP"=state_fips_code, acre_feet_applied)

ac_per_sqm <- 1 / 4046.86

state_volume_sf <- nass_data(
  source_desc="CENSUS",
  short_desc="AG LAND, IRRIGATED - ACRES",
  agg_level_desc="STATE",
  year="2017",
  domain_desc="TOTAL") %>%
select("STATEFP"=state_fips_code, acres_irrigated) %>%
full_join(state_volume) %>%
right_join(state_sf) %>%
st_as_sf %>%
mutate(
  irrigated_area_ac=exact_extract(irrigated_areas, state_volume_sf, 'sum') * prod(res(irrigated_areas)) * ac_per_sqm
  acre_feet_per_acre=acre_feet_applied
  ) %>%
write_sf(output_file)
