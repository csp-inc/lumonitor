import geopandas as gpd
from pystac_client import Client
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")

aoi = gpd.read_file('data/azml/conus.geojson')

time_range = "2016-01-01/2016-12-31"
search = catalog.search(
    collections=['landsat-8-c2-l2'],
    intersects=aoi.to_dict()['geometry'][0],
    datetime=time_range
)
print(f"{search.matched()} Items found")
