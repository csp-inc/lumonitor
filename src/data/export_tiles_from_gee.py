import argparse
import os

import ee
import fsspec
import rasterio as rio

ee.Initialize()

parser = argparse.ArgumentParser()
parser.add_argument('--account-name')
parser.add_argument('--account-key')

args = parser.parse_args()

os.environ['AZURE_STORAGE_ACCOUNT'] = args.account_name
os.environ['AZURE_STORAGE_ACCESS_KEY'] = args.account_key

def export_training_data(raster, output_prefix):
    cells = raster.profile['width']

    bounds = list(raster.bounds)
    epsg_string = f'EPSG:{raster.crs.to_epsg()}'
    projection = ee.Projection(epsg_string)
    tile = ee.Geometry.Rectangle(bounds, projection, False)

    hm = ee.Image('projects/GEE_CSP/HM/HM_ee_2017_v014_500_30').multiply(10000).int16()

    nlcd = ee.Image('USGS/NLCD/NLCD2016')
    nlcd_imp_d = nlcd.select('impervious_descriptor').int16()
    nlcd_imp = nlcd.select('impervious').int16()

    hm = hm.addBands(nlcd_imp_d)
    hm = hm.addBands(nlcd_imp)

    task = ee.batch.Export.image.toCloudStorage(
        image=hm,
        bucket='lumonitor',
        fileNamePrefix=output_prefix,
        dimensions=cells,
        region=tile,
        crs=epsg_string
        )

    task.start()


def get_label_for_trainer(training_cog):
    _, _, year, _, cog_file = training_cog.split('/')
    return f'lumonitor/cog/{year}/training/l{cog_file}'


def get_labels_for_trainers(training_cogs):
    return set([get_label_for_trainer(c) for c in training_cogs])


def get_trainer_for_label(label_cog):
    _, _, year, _, file = label_cog.split('/')
    training_file = file[1:]
    return f'/vsiaz/hls/cog/{year}/training/{training_file}'


az_fs = fsspec.filesystem(
    'az',
    account_name=args.account_name,
    account_key=args.account_key
)

gcs_fs = fsspec.filesystem('gcs')


training_cogs = set(az_fs.find('hls/cog'))
label_cogs_for_training_cogs = get_labels_for_trainers(training_cogs)
label_cogs = set(gcs_fs.find('lumonitor/cog'))

label_cogs_to_run = label_cogs_for_training_cogs - label_cogs

for output_cog in label_cogs_to_run:
    training_path = get_trainer_for_label(output_cog)
    print(training_path)
    training_rio = rio.open(training_path)
    export_training_data(training_rio, output_cog)
