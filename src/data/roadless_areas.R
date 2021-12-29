library(sf)

iras <- read_sf("/vsizip/vsicurl/https://data.fs.usda.gov/geodata/edw/edw_resources/shp/S_USA.RoadlessArea_2001.zip") %>%
  summarise() %>%
  st_transform(4326)

load_blm_layer <- function(name) {
  # https://gis.blm.gov/EGISDownload/LayerPackages/BLM_National_NLCS_Wilderness_and_WildernessStudyAreas_poly.zip
  blm_file <- "data/BLM_National_NLCS_Wilderness_and_WildernessStudyAreas_poly/nlcs.gdb"
  read_sf(blm_file, name) %>%
    st_cast("MULTIPOLYGON") %>%
    summarise() %>%
    st_transform(4326) %>%
    rename(geometry = Shape)
}

blm <- rbind(load_blm_layer("nlcs_wld_poly"), load_blm_layer("nlcs_wsa_poly"))

rbind(iras, blm) %>%
  summarise() %>%
  write_sf("data/azml/roadless_areas.gpkg")
