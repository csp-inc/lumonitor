import geopandas as gpd
import rasterio as rio
import pygeos


def range_with_end(start: int, end: int, step: int) -> int:
    i = start
    while i < end:
        yield i
        i += step
    yield end


x_min = 1751329.01635844
y_min = 2399033.11023793
x_max = 1932169.68463254
y_max = 2710550.69480397

coords = [
    (x, y)
    for x in range_with_end(
        x_min,
        x_max - 512,
        70
    )
    for y in range_with_end(
        y_min,
        y_max - 512,
        70
    )
]

points = pygeos.creation.points(coords)

crs = rio.open('data/azml/conus_hls_median_2016.vrt').crs
aoi = pygeos.io.from_shapely(gpd.read_file('data/azml/conus.geojson').to_crs(crs))

pygeos.prepare(aoi)
pygeos.prepare(points)

points_in_aoi = pygeos.contains(aoi, points)
