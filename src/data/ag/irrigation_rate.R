library(sf)
library(dplyr)
library(tidyr)
library(tigris)
library(usdarnass)

#output_file <- commandArgs(TRUE)[1]
state_sf <- states(cb=TRUE, resolution="5m", year=2018)

state_volume <- nass_data(
  agg_level_desc="STATE", 
  year="2018", 
  short_desc="AG LAND, IRRIGATED - WATER APPLIED, MEASURED IN ACRE FEET", 
  domain_desc="TOTAL") %>%
mutate(acre_feet_applied = as.numeric(gsub(',', '', Value))) %>%
select("STATEFP"=state_fips_code, acre_feet_applied)

nass_data(
  source_desc="CENSUS", 
  short_desc="AG LAND, IRRIGATED - ACRES", 
  agg_level_desc="STATE", 
  year="2017", 
  domain_desc="TOTAL") %>%
mutate(acres_irrigated = replace_na(as.numeric(gsub(',', '', Value)), 0)) %>%
select("STATEFP"=state_fips_code, acres_irrigated) %>%
full_join(state_volume) %>%
right_join(state_sf) %>%
mutate(
  acre_feet_per_acre_irrigated=acre_feet_applied / acres_irrigated,
  id = 1:n()
) %>%
select(id, acre_feet_per_acre_irrigated, geometry) %>%
st_as_sf %>%
st_transform(4326) %>%
write_sf(output_file)
