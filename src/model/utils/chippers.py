from typing import List

from numpy import ndarray
from numpy.ma import masked_array


def nlcd_chipper(chip_raw: ndarray) -> ndarray:
    # Don't know where these -1000s are coming from, but
    # they are there on read
    masked = (chip_raw > 100) | (chip_raw == -1000)
    # Replace nodatas with 0,
    # then divide by 100 for real values
    return (
        masked_array(
            chip_raw,
            mask=masked,
        ).filled(0)
        / 100.0
    )


def hm_chipper(chip_raw: ndarray) -> ndarray:
    masked = chip_raw == -32768
    return masked_array(chip_raw, mask=masked).filled(0) / 10000.0
