library(raster)
library(grDevices)

args <- commandArgs(TRUE)
input_file <- args[1]
output_file <- args[2]

# Can't have zeroes b/c they go nodata.
# I set it up this way for another project so they would import correctly
# to mapbox.  May be able to mitigate by changing the datatype below.
colors <- c('#010101', '#ff0101', '#ffbb01', '#ffff01')

n_colors <- 10
ramp <- colorRampPalette(colors)(n_colors)
breaks <- seq(0, 1, length.out=n_colors + 1)

input_raster <- raster(input_file) 
output_raster <- RGB(input_raster, col=ramp, breaks=breaks, colNA='black')

options = c("COMPRESS=LZW", "BIGTIFF=YES", "TILED=TRUE", "BLOCKXSIZE=256",      
            "BLOCKYSIZE=256") 

writeRaster(
  output_raster,          
  filename=output_file,
  datatype='INT1U',
  NAflag=0,
  overwrite=TRUE,
  options=options
)
