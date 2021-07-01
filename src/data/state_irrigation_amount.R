library(sf)
library(dplyr)
library(tigris)
library(usdarnass)

nass_data(
  agg_level_desc="STATE", 
  year="2018", 
  short_desc="AG LAND, IRRIGATED - WATER APPLIED, MEASURED IN ACRE FEET", 
  domain_desc="TOTAL"
) %>%
select("STATEFP"=state_fips_code, Value) %>% 
mutate(Value = as.numeric(gsub(',', '', Value))) %>%
right_join(states(cb=TRUE, resolution="5m", year=2018)) %>% 
mutate(Value = Value / ALAND) %>%
st_as_sf %>%
write_sf('~/Desktop/irrigation_volume.gpkg')
