import os

from azure.storage.blob import ContainerClient

client = ContainerClient(
    account_url='https://usfs.blob.core.windows.net',
    container_name='app',
    credential=os.environ['AZURE_USFS_STORAGE_KEY']
)

for blob in client.list_blobs('tiles_CONUS_fire_bounds_2016_2018'):
    if blob.name.endswith('pbf'):
        print(blob.name)
        blob.content_settings.content_type = "application/vnd.mapbox-vector-tile"
        blob.content_settings.content_encoding = "gzip"
        client.get_blob_client(blob).set_http_headers(blob.content_settings)
