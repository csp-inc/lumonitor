# I save as a csv with column names 'lon' and 'lat' for possible future 
# incorporation into @ohiat's azure based workflow 
library(readr)
library(sf)
set.seed(1337)

args <- commandArgs(trailingOnly=TRUE)

input_spatial_file <- args[1]
output_csv <- args[2]

sample_area <- read_sf(input_spatial_file)

# Or whatever
number_of_points <- 1e5
points <- st_sample(sample_area, number_of_points)

coords_df <- data.frame(st_coordinates(points))
names(coords_df) <- c('lon', 'lat')
coords_df$ID <- 1:nrow(coords_df)

write_csv(coords_df, output_csv)
