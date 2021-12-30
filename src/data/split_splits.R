library(dplyr)
library(sf)

unsplit_sf <- read_sf("data/azml/conus_projected.gpkg")
split_sf <- read_sf("conus_split_204.shp") %>%
  group_by(POLY_ID) %>%
  summarise() %>%
  st_cast("MULTIPOLYGON")

st_crs(split_sf) <- st_crs(unsplit_sf)

for (id in split_sf$POLY_ID) {
  this_sf <- split_sf[split_sf$POLY_ID == id, ]
  this_f <- paste0("data/azml/conus_split_20/", as.character(id), ".gpkg")
  write_sf(this_sf, this_f)
}
