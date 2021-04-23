import os

import fsspec

ACCOUNT_NAME = 'usfs'
ACCOUNT_KEY = os.environ['AZURE_USFS_STORAGE_KEY']

fs = fsspec.filesystem(
    'az',
    account_name=ACCOUNT_NAME,
    account_key=ACCOUNT_KEY
)

for path in fs.find('app/tiles_CONUS_fire_bounds_2016_2018'):
    if path.endswith('.pbf'):
        # 'app' is prepended to the path, this removes it
        blob_path = os.path.relpath(path, 'app')
        os.system(f'az storage blob update --account-name {ACCOUNT_NAME} --account-key {ACCOUNT_KEY} --container app --name {blob_path} --content-type application/vnd.mapbox-vector-tile --content-encoding gzip')

