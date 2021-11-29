from pyproj import CRS
import rioxarray as rx
import xarray as xr

from dask.distributed import Client, Lock


if __name__ == "__main__":
    #    client = Client()
    #    print(client.dashboard_link)

    out_crs = rx.open_rasterio("data/irrigated_areas.tif").rio.crs

    da = xr.open_zarr(
        "https://daymeteuwest.blob.core.windows.net/daymet-zarr/annual/na.zarr",
        consolidated=True,
    )

    proj = CRS.from_cf(da.lambert_conformal_conic.to_dict()["attrs"])
    year = [l[0:4] for l in da.time.values.astype(str).tolist()]
    prcp = da["prcp"].assign_coords(time=year).to_dataset("time")

    for target_year in ["2013", "2016", "2020"]:
        prcp[target_year].drop(["lon", "lat"]).rio.write_crs(proj).rio.reproject(
            out_crs
        ).rio.to_raster(
            f"data/daymet_precip_{target_year}.tif",
            dtype="float32",
            compress="LZW",
            predictor=3,
            tiled=True,
            crs=out_crs,
        )
