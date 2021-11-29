from functools import partial

import geocube
import geopandas as gpd
import numpy as np
import rioxarray as rx
import xarray as xr
from dask.distributed import Client, Lock
from geocube.api.core import make_geocube


def zonal_stats(df: gpd.GeoDataFrame, zone_col: str, value_da: str, like: xr.Dataset):

    def zonal_stats_block(block):
        zones = make_geocube(
            vector_data=df,
            measurements=[zone_col],
            like=block,
            fill=np.nan,
            rasterize_function=partial(geocube.rasterize.rasterize_image, all_touched=True)
        ).assign_coords(block.coords)
        zones[value_da] = block[value_da]
#        zones['urban_impact'] = block.urban_impact
        # groupby complains if zone_col is all nan within the block
        if np.isnan(zones.id).all().item(0):
            return zones
        return zones

    return like.map_blocks(
        zonal_stats_block,
        template=like
    ).drop('spatial_ref')


if __name__ == '__main__':
    client = Client()
    print(client.dashboard_link)

    states = gpd.read_file('ag1.gpkg')

    values_da = rx.open_rasterio(
        'ag1.tif',
#        'data/conus_2020_prediction_4326.tif',
        chunks={'x': 4096, 'y': 4096},
        lock=False
    ).squeeze().drop('band')
    values = values_da.to_dataset(**{'name': 'id'})
    values['urban_impact'] = values_da
    
    # This doesn't work because groupby brings it all into memory (see various
    # gh issues, etc about this
    stats = zonal_stats(states, 'id', 'urban_impact', values).groupby('id').mean()

    stats.to_dataframe().to_csv('test.csv')
