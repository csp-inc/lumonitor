import ee
from numpy import arange

ee.Initialize()


def edge_effect(img: ee.Image) -> ee.Image:
    ones = ee.Image(1)
    resolutionOut = 30
    # Set some constants
    distance = min(255 * resolutionOut, 10000)
    halfLife = 500
    HMeeMax = ee.Image(0)

    seq = arange(0.1, 1.05, 0.05)
    for i in seq:
        HMt = img.gt(i)
        # dt: filter out small "salt" patches
        # JA: keeping this at 90 for now
        HMt = HMt.focal_mean(90, "circle", "meters")
        HMt = HMt.gt(0.5)

        # dt: calculate Euc distance away from 1s
        HMtdistance = HMt.distance(ee.Kernel.euclidean(distance, "meters"), False)
        HMee = (
            ee.Image(i)
            .multiply(ee.Image(0.5).pow(HMtdistance.divide(ee.Image(halfLife))))
            .unmask(0)
        )
        HMeeMax = HMeeMax.max(HMee)

    return ones.subtract(ones.subtract(HMeeMax).multiply(ones.subtract(img)))
