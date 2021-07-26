import dask
from dask.distributed import Client
import dask_geomodeling
import rioxarray

c = Client()

geom = dask_geomodeling.geometry.sources.GeometryFileSource('data/irrigation_rate.gpkg')

r = dask_geomodeling.raster.misc.Rasterize(geom, 'acre_feet_per_acre_irrigated')

rtemplate = rioxarray.open_rasterio('data/irrigated_areas.tif')

request = {'mode':'vals', 'bbox':rtemplate.rio.bounds(), 'width':rtemplate.rio.width, 'height':rtemplate.rio.height, 'projection':rtemplate.rio.crs.to_wkt()}

g, name = r.get_compute_graph(**request)

# Unfortunately too large and can't figure out how to write it
result = dask.get(g, [name])




