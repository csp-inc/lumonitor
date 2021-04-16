export CPL_TMPDIR=/mnt
export AZURE_STORAGE_ACCOUNT=lumonitoreastus2
export AZURE_STORAGE_ACCESS_KEY=i3sq5daNz09EgYfksKkbxBnfSh1ngyVHRvnZWu4/KVtJWq72krfHttryoyxbAkmoE56W6xQMmsnvVrnAsuKsVA==
export CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE=YES
gdal_merge.py -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co COMPRESS=LZW -co PREDICTOR=2 -co BIGTIFF=YES -o /vsiaz/hls/conus_median_2016.tif -ps 30 30 -ul_lr -2493045.000 3310005.000 2342655.000 177285.000 $(az storage blob list --prefix=cog/2016/training_albers -c 'hls' --account-name=lumonitoreastus2 --account-key=i3sq5daNz09EgYfksKkbxBnfSh1ngyVHRvnZWu4/KVtJWq72krfHttryoyxbAkmoE56W6xQMmsnvVrnAsuKsVA== --query []["name"] -o tsv | sed 's/^/\/vsiaz\/hls\//')

gdalbuildvrt -te -2493045.00 2342655.000 177285.000 3310005.000 -overwrite cog_mosaic.vrt $(az storage blob list --prefix=cog/2016/training_albers -c 'hls' --account-name=lumonitoreastus2 --account-key=i3sq5daNz09EgYfksKkbxBnfSh1ngyVHRvnZWu4/KVtJWq72krfHttryoyxbAkmoE56W6xQMmsnvVrnAsuKsVA== --query []["name"] -o tsv | sed 's/^/\/vsiaz\/hls\//')

