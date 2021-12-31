import argparse
from osgeo import gdal

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file")
    args = parser.parse_args()

    ds = gdal.Open(args.file)
    names = [
        "Combined Impacts",
        "Agricultural Impacts",
        "Transportation Impacts",
        "Urban Impacts",
    ]
    for i in range(4):
        b = ds.GetRasterBand(i + 1)
        b.SetDescription(names[i])
        b.SetScale(0.01)

    ds = None
