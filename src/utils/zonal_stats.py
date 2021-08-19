from functools import partial

import geocube
import geopandas as gpd
import numpy as np
import rioxarray as rx
import xarray as xr
from dask.distributed import Client, Lock
from geocube.api.core import make_geocube


def zonal_stats(
        df: gpd.GeoDataFrame,
        zone_col: str,
        like
):

    def zonal_stats_block(block):
        zones = make_geocube(
            vector_data=df,
            measurements=[zone_col],
            like=block,
            fill=np.nan,
            rasterize_function=partial(geocube.rasterize.rasterize_image, all_touched=True)
        ).assign_coords(block.coords)
        zones['urban_impact'] = block.urban_impact
        # groupby complains if zone_col is all nan within the block
        if np.isnan(zones.id).all().item(0):
            return zones
        return zones.groupby(zone_col)

    return like.map_blocks(
        zonal_stats_block,
        template=like
    ).drop('spatial_ref')


if __name__ == '__main__':
    client = Client()
    print(client.dashboard_link)

    states = gpd.read_file('ag1.gpkg')#'data/irrigation_rate.gpkg')

    # Need to format this as a dataset so map_blocks won't complain
    # when it comes back
    values_da = rx.open_rasterio(
        'ag1.tif',
#        'data/conus_2020_prediction_4326.tif',
        chunks={'x': 4096, 'y': 4096},
        lock=False
    ).squeeze().drop('band')
    values = values_da.to_dataset(**{'name': 'id'})
    values['urban_impact'] = values_da
    
    stats = zonal_stats(states, 'id', values).mean()
    print(stats)

    stats.to_dataframe().to_csv('test.csv')
