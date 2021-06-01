## Lumonitor

Lumonitor is short for "land use monitor", but we are really only interested in one thing: human impacts. Our current phase of work is modeling the intensity of human impacts using the 2016 NLCD Impervious Surface layer and Harmonized Landsat Sentinel data. The majority of work so far has been in identifying and/or building the computational infrastructure (via the Azure platform) necessary to produce the model output, e.g. a continential-level prediction of current human impacts. This project is being funded by the Microsoft AI for Earth Program.

## The model

Our goal is to train a UNet-type autoencoder on the 30-m 2016 NLCD Impervious Surface map and Landsat data from the Harmonized Landsat sentinel. For training data, we are using the cloud-free annual median for each pixel in the seven available bands.

## The workflow

Our current workflow is described in detail below. We are essentially modeling our process based on this notebook outlining a similar workflow using Google Earth Engine, etc.

1. Calculate 2016 HLS Cloud-Free median values
The HLS data are available on Azure Open Datasets. Upon initiation of the project there was no STAC or other catalog available. Loading each image within a single year even for a single tile consumed too much memory, so we had to identify a process which did not require this, essentially eliminating a GDAL / rasterio / numpy type approach. (One possibility which was not tried was to use the rasterio.windows module to read portions of each image in an iterative fashion, though that would like take too much time.) The approach we eventually settled on was to deploy pangeo on a daskhub instance running on Azure Kubernetes Service. Using xarray/dask for calculating the medians took approximately 2 minutes per tile (for each of approximately 1000 tiles spanning the conterminous US in each year). 
A major issue, brought to the forefront in later steps, was that zarr data was not the ideal solution for storing the data (originally sourced in COG). The primary issue was slow read time in the data loading and modeling steps, but lack of mature spatial support was also an issue.

2. Sample
We training initial models in tensorflow but soon moved to pytorch. One goal of our setup was to dynamically sample the training data directly from Blob Storage while running the model. In implementing our workflow, we determined the best performance would be gained by reading from COG format using the rasterio.windows module. As mentioned above, our training data were in zarr, so we needed to convert. Unfortunately writing directly to Blob Storage using rioxarray/rasterio was error prone and unsupported. After some back and forth with rasterio developers, our only realistic option was to fork rasterio and implement the necessary changes. This required us to download all zarrs and re-export as COGs.

3. Model
We used pytorch and azureml to train our models. We spent the typical amount of time needed to test different models, etc. Some features of our process (besides dynamically sampling from COG as mentioned above) include a "dual-function" dataset which pulls random tiles, but also gridded data. Each can also be pulled within a pre-defined area stored in vector file. Because the training data was heavily zero-inflated, training and test loss were typically extremely low. However, model output was excellent in most cases. To further evaluate a running model, we developed a process to predict and symbolize the results over small pre-defined test areas at each epoch, allowing using to evaluate model efficacy in real time beyond our loss statistics.

5. Predict
The use of a UNet-type model with many convolutions meant that for a typical 512x512 "chip", only the center 70x70 pixel area was unaffected by padding/cropping. For continental-level prediction, this meant we had to pull and evaluate millions of chips. We strugged to identify an ideal platform for this process, and are currently using up to 20 K80 GPU instances running in azure-ml to produce results in a set number of "regions" (typically 100) across the area of interest. This process currently takes about 8 hours per year - a reasonable time.

7. Mosaic and Export
Ideally regional predictions could be mosaicked as part of the prediction process, but a number of issues have prevented us from completely implementing this process. We currently pull predictions for each region to a local machine and mosaic using gdal_merge.py and gdalwarp. This takes several hours but is not time prohibitive. We prepare symbolized and tiled raster data for deployment on a web application using matplotlib/numpy (or the R raster package) and gdal.

## Next steps

Eventually our model will include all human impacts (not just urban), and we our hopeful our time spent developing the workflow for initial results will prove fruitful in accelerating ongoing development.

With the advent of the Planetary Computer, some data engineering and cataloguing issues described above will likely be mitigated.

