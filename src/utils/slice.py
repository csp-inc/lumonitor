import math

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import box


def slice(area_file: str, n: int, output_file_template: str) -> None:
    # output_file_template is a no-f fstring e.g. 'data/i_{n}.shp'
    areas = gpd.read_file(area_file)
    xmin, ymin, xmax, ymax = areas.total_bounds
    width = xmax - xmin
    height = ymax - ymin
    side_length = math.sqrt(height * width / n)
    n_cols = math.ceil(width / side_length)

    for this_n in range(n):
        this_row = this_n // n_cols
        this_col = this_n % n_cols
        i_xmin = xmin + this_col * side_length
        i_ymin = ymin + this_row * side_length
        i_xmax = i_xmin + side_length
        i_ymax = i_ymin + side_length
        this_box = GeoDataFrame(
            geometry=[box(i_xmin, i_ymin, i_xmax, i_ymax)],
            crs=areas.crs
        )
        this_area = gpd.overlay(areas, this_box, how='intersection')
        if len(this_area.index) > 0:
            output_file = output_file_template.format(n=this_n)
            print(output_file)
            this_area.to_file(output_file, driver='GeoJSON')


area_file = 'data/azml/conus.geojson'
n = 100
output_file_template = 'data/azml/slices/conus_{n}.geojson'

slice(area_file, n, output_file_template)
