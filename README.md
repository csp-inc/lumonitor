## Lumonitor

Lumonitor is short for "land use monitor", but we are really only interested in
one thing: human impacts. For this project we modeled the intensity of human
impacts using a CSP dataset on Anthropogenic Impacts and Landsat 8 data from the
Harmonized Landsat Sentinel dataset. Model results are viewable
[here](https://cspbeta.z22.web.core.windows.net/project2). This project was
generously funded by the Microsoft AI for Earth Program.

## The model

Our goal was to train a UNet-type autoencoder on the 30-m Human Modification
dataset from the [Disappearing West](https://www.disappearingwest.org)/U.S.
project and Landsat 8 data from the Harmonized Landsat sentinel. For training
data, we used the cloud-free annual median for each pixel in the seven
available bands.

## The workflow

### 1. Training data

We used cloud-free annual per-pixel median values of the [Harmonized Landsat
Sentinel](https://hls.gsfc.nasa.gov/) dataset and various label data to train
our model. The HLS data are available on [Azure Open
Datasets](https://github.com/microsoft/AIforEarthDataSets/blob/main/data/hls.md).
Upon initiation of the project there was no STAC or other catalog available
(this need is now filled by the [Planetary
Computer](https://innovation.microsoft.com/en-us/planetary-computer) for many
datasets). However, loading each image within a single year even for a single
3660x3660 pixel "tile" consumed too much memory, essentially eliminating a GDAL
/ rasterio / numpy type approach. The approach we eventually settled on was to
deploy pangeo on a daskhub instance running on [Azure Kubernetes
Service](https://azure.microsoft.com/en-us/services/kubernetes-service/). Using
xarray/dask for calculating the medians took approximately 2 minutes per tile
(for each of approximately 1000 tiles spanning the conterminous US in each
year). Code for this step is available
[here](https://github.com/csp-inc/data-ingestion/tree/lumonitor).

Label data were exported from Google Earth Engine and storage on Azure Blob
Storage using [this script](src/data/export_hm.py).

One goal of our setup was to dynamically sample the training data directly from
Blob Storage while running the model. In implementing our workflow, we
determined the best performance would be gained by reading from COG format using
the rasterio.windows module. Training data were initially exported in zarr
format (since that is the only format which supports writing directly to Azure
Blob Storage from xarray), so we downloaded and converted all files, then
reuploaded to Blob Storage, using [this script](src/utils/zarr_to_cog.py).

### 2. Model Training
We used Pytorch and Azure ML to train our models. We spent the typical amount of
time needed to test different models, etc. Some features of our process (besides
dynamically sampling from COG as mentioned above) include a "dual-function"
dataset which pulls random chips, but also gridded data representing all chips
for a predefined area. Each can also be pulled within a specific area stored in
a shapefile etc. Because the training data was heavily zero-inflated, training
and test loss were typically extremely low. However, model output was excellent
in most cases. But to further evaluate a running model, we developed a process
to predict and symbolize the results over small pre-defined test areas at each
epoch, allowing using to evaluate model efficacy in real time beyond our loss
statistics. Model training was accomplished using
[src/model/run_training.py](src/model/run_training.py) and
[src/model/train.py](src/model/train.py).

### 3. Prediction
The use of a UNet-type model with many convolutions meant that for a typical
512x512 "chip", only the center 70x70 pixel area was unaffected by
padding/cropping. For continental-level prediction, this meant we had to pull
and evaluate millions of chips. We used Azure ML and up to 20 K80 GPU instances
to produce results in a set number of "regions" (typically 100) across the area
of interest. This process currently takes about 8 hours per year of data - a
reasonable time. Prediction code files are
[src/model/run_prediction.py](src/model/run_prediction.py) and
[src/model/predict.py](src/model/predict.py).

### 4. Data Export
