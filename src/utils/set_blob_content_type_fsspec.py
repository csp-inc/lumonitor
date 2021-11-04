# this is slower
import os

import fsspec

ACCOUNT_NAME = "lumonitor"
ACCOUNT_KEY = os.environ["AZURE_LUMONITOR_STORAGE_KEY"]

fs = fsspec.filesystem("az", account_name=ACCOUNT_NAME, account_key=ACCOUNT_KEY)

for path in fs.find("tiles/summary_stats"):
    if path.endswith(".pbf"):
        # 'app' is prepended to the path, this removes it
        blob_path = os.path.relpath(path, "tiles")
        print(blob_path)
        os.system(
            f"az storage blob update --account-name {ACCOUNT_NAME} --account-key {ACCOUNT_KEY} --container tiles --name {blob_path} --content-type application/vnd.mapbox-vector-tile --content-encoding gzip"
        )
