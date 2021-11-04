import os

from azure.storage.blob import ContainerClient

client = ContainerClient(
    account_url="https://lumonitor.blob.core.windows.net",
    container_name="tiles",
    credential=os.environ["AZURE_LUMONITOR_STORAGE_KEY"],
)

for blob in client.list_blobs("summary_stats"):
    if blob.name.endswith("pbf"):
        print(blob.name)
        blob.content_settings.content_type = "application/vnd.mapbox-vector-tile"
        blob.content_settings.content_encoding = "gzip"
        client.get_blob_client(blob).set_http_headers(blob.content_settings)
