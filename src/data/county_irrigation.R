library(sf)
library(dplyr)
library(tigris)
library(usdarnass)

nass_data(
  source_desc="CENSUS", 
  short_desc="AG LAND, IRRIGATED - ACRES", 
  agg_level_desc="COUNTY", 
  year="2017", 
  domain_desc="TOTAL") %>%
select("STATEFP"=state_fips_code, "COUNTYFP"=county_code, Value) %>% 
mutate(Value = as.numeric(gsub(',', '', Value))) %>%
right_join(counties()) %>% 
st_as_sf %>%
write_sf('~/Desktop/irrigation.gpkg')
