import ee

from export_hm import export_hm
from edge_effect import edge_effect

ee.Initialize()

hurban = ee.Image("projects/GEE_CSP/HM/alternate_versions/HM_2017v012c_30_75th").select(
    "hurban"
)

hurban_ee = edge_effect(hurban)

export_hm(hurban_ee, scale=30, output_prefix="Hurban_2017v012c_30_75th", run_task=True)
