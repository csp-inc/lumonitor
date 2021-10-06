import geopandas as gpd
import io
import pandas as pd
import rtree
import requests
from shapely.geometry import box

class HLSTileLookup:
    """Wrapper around an rtree for finding HLS tile ids."""

    def __init__(self):
        self.tile_extents = self._get_extents()
        self.tile_gpd = self._get_boxes()
        self.tree_idx = rtree.index.Index()
        self.idx_to_id = {}
        self.idx_to_all = {}
        for i_row, row in self.tile_extents.iterrows():
            self.tree_idx.insert(i_row, (row.MinLon, row.MinLat, row.MaxLon, row.MaxLat))
            self.idx_to_id[i_row] = row.TilID

    def _get_extents(self):
        hls_tile_extents_url = 'https://ai4edatasetspublicassets.blob.core.windows.net/assets/S2_TilingSystem2-1.txt?st=2019-08-23T03%3A25%3A57Z&se=2028-08-24T03%3A25%3A00Z&sp=rl&sv=2018-03-28&sr=b&sig=KHNZHIJuVG2KqwpnlsJ8truIT5saih8KrVj3f45ABKY%3D'
        # Load this file into a table, where each row is:
        # Tile ID, Xstart, Ystart, UZ, EPSG, MinLon, MaxLon, MinLat, MaxLat
        print('Reading tile extents...')
        s = requests.get(hls_tile_extents_url).content
        hls_tile_extents = pd.read_csv(io.StringIO(s.decode('utf-8')),delimiter=r'\s+')
        print('Read tile extents for {} tiles'.format(len(hls_tile_extents)))
        return hls_tile_extents

    def _get_boxes(self):
        return gpd.GeoDataFrame(
            self.tile_extents,
            crs="EPSG:4326",
            geometry=[
                box(r['MinLon'], r['MinLat'], r['MaxLon'], r['MaxLat'])
                for _, r in self.tile_extents.iterrows()
            ]
        )

    def get_point_hls_tile_ids(self, lat, lon):
        results = list(self.tree_idx.intersection((lon, lat, lon, lat)))
        return set([self.idx_to_id[r] for r in results])

    def get_bbox_hls_tile_ids(self, left, bottom, right, top):
        return set(
            self.idx_to_id[i]
            for i in self.tree_idx.intersection((left, bottom, right, top))
        )

    def get_geometry_hls_tile_ids(self, geometry):
        tiles_in_aoi = gpd.overlay(geometry, self.tile_gpd, how='intersection')
        return set(tiles_in_aoi['TilID'])

    def get_hls_tile_info(self, tile_ids):
        return self.tile_extents[self.tile_extents['TilID'].isin(tile_ids)]

    def get_point_hls_tile_info(self, lat, lon):
        tile_ids = self.get_point_hls_tile_ids(lat, lon)
        return self.get_hls_tile_info(tile_ids)

    def get_bbox_hls_tile_info(self, left, bottom, right, top):
        tile_ids = self.get_bbox_hls_tile_ids(left, bottom, right, top)
        return self.get_hls_tile_info(tile_ids)

    def get_geometry_hls_tile_info(self, geometry):
        tile_ids = self.get_geometry_hls_tile_ids(geometry)
        return self.get_hls_tile_info(tile_ids)
