SHELL=/usr/bin/env bash
PREDICTION_DIR=data/predictions/
OUTPUT_DIR=data/output/
VPATH=data/:$(PREDICTION_DIR):$(OUTPUT_DIR)


AG_PREFIX=a
AG_RUNID=hm-2016_1640031699_24788cc5
AG_2013=$(PREDICTION_DIR)$(AG_PREFIX)_13_$(AG_RUNID)/$(AG_PREFIX)_13_prediction.tif
AG_2016=$(PREDICTION_DIR)$(AG_PREFIX)_16_$(AG_RUNID)/$(AG_PREFIX)_16_prediction.tif
AG_2020=$(PREDICTION_DIR)$(AG_PREFIX)_20_$(AG_RUNID)/$(AG_PREFIX)_20_prediction.tif

TRANS_PREFIX=t
TRANS_RUNID=hm-2016_1639946057_b79ecb57
TRANS_2013=$(PREDICTION_DIR)$(TRANS_PREFIX)_13_$(TRANS_RUNID)/$(TRANS_PREFIX)_13_prediction.tif
TRANS_2016=$(PREDICTION_DIR)$(TRANS_PREFIX)_16_$(TRANS_RUNID)/$(TRANS_PREFIX)_16_prediction.tif
TRANS_2020=$(PREDICTION_DIR)$(TRANS_PREFIX)_20_$(TRANS_RUNID)/$(TRANS_PREFIX)_20_prediction.tif

URBAN_PREFIX=u
URBAN_RUNID=hm-2016_1640024836_f7c93937
URBAN_2013=$(PREDICTION_DIR)$(URBAN_PREFIX)_13_$(URBAN_RUNID)/$(URBAN_PREFIX)_13_prediction.tif
URBAN_2016=$(PREDICTION_DIR)$(URBAN_PREFIX)_16_$(URBAN_RUNID)/$(URBAN_PREFIX)_16_prediction.tif
URBAN_2020=$(PREDICTION_DIR)$(URBAN_PREFIX)_20_$(URBAN_RUNID)/$(URBAN_PREFIX)_20_prediction.tif

LAYERS=ag trans urban
BLAYERS=at tu au atu

YEARS=2013 2016 2020
LAYER_YEARS=$(foreach LAYER, ${LAYERS}, $(foreach YEAR, ${YEARS}, ${LAYER}_${YEAR}))
BLAYER_YEARS=$(foreach BLAYER, ${BLAYERS}, $(foreach YEAR, ${YEARS}, ${BLAYER}_${YEAR}_blend))
RGBS=$(patsubst %, $(OUTPUT_DIR)%_rgb.tif, $(LAYER_YEARS) $(BLAYER_YEARS))
TILES=$(patsubst %, %.tiles, $(LAYER_YEARS) $(BLAYER_YEARS))

tiles: $(TILES)
rgbs: $(RGBS)

%_blend_rgb.tif:
	source src/utils/blend_images.sh $^ $@

%_rgb.tif:
	source src/utils/export_rgb.sh $^ $@

%.tiles: %_rgb.tif
	source src/utils/make_raster_tiles.sh $^
	touch data/$@

# NOt the prettiest, I admit
$(OUTPUT_DIR)at_2013_blend_rgb.tif: ag_2013_rgb.tif trans_2013_rgb.tif
$(OUTPUT_DIR)at_2016_blend_rgb.tif: ag_2016_rgb.tif trans_2016_rgb.tif
$(OUTPUT_DIR)at_2020_blend_rgb.tif: ag_2020_rgb.tif trans_2020_rgb.tif
$(OUTPUT_DIR)au_2013_blend_rgb.tif: ag_2013_rgb.tif urban_2013_rgb.tif
$(OUTPUT_DIR)au_2016_blend_rgb.tif: ag_2016_rgb.tif urban_2016_rgb.tif
$(OUTPUT_DIR)au_2020_blend_rgb.tif: ag_2020_rgb.tif urban_2020_rgb.tif
$(OUTPUT_DIR)tu_2013_blend_rgb.tif: trans_2013_rgb.tif urban_2013_rgb.tif
$(OUTPUT_DIR)tu_2016_blend_rgb.tif: trans_2016_rgb.tif urban_2016_rgb.tif
$(OUTPUT_DIR)tu_2020_blend_rgb.tif: trans_2020_rgb.tif urban_2020_rgb.tif
$(OUTPUT_DIR)atu_2013_blend_rgb.tif: tu_2013_blend_rgb.tif ag_2013_rgb.tif
$(OUTPUT_DIR)atu_2016_blend_rgb.tif: tu_2016_blend_rgb.tif ag_2016_rgb.tif
$(OUTPUT_DIR)atu_2020_blend_rgb.tif: tu_2020_blend_rgb.tif ag_2020_rgb.tif

# For ag, just use what the model returns
$(OUTPUT_DIR)ag_2013_rgb.tif: $(AG_2013) ag_col.txt
$(OUTPUT_DIR)ag_2016_rgb.tif: $(AG_2016) ag_col.txt
$(OUTPUT_DIR)ag_2020_rgb.tif: $(AG_2020) ag_col.txt

# For trans and urban, do min calcs for 2013 and 2016
$(OUTPUT_DIR)trans_2020_rgb.tif: $(TRANS_2020) trans_col.txt
$(PREDICTION_DIR)trans_2016_min.tif: $(TRANS_2020) $(TRANS_2016)
	source src/utils/min_calc.sh $^ $@
$(OUTPUT_DIR)trans_2016_rgb.tif: trans_2016_min.tif trans_col.txt
$(PREDICTION_DIR)trans_2013_min.tif: $(TRANS_2013) trans_2016_min.tif
	source src/utils/min_calc.sh $^ $@
$(OUTPUT_DIR)trans_2013_rgb.tif: trans_2013_min.tif trans_col.txt

$(OUTPUT_DIR)urban_2020_rgb.tif: $(URBAN_2020) urban_col.txt
$(PREDICTION_DIR)urban_2016_min.tif: $(URBAN_2020) $(URBAN_2016)
	source src/utils/min_calc.sh $^ $@
$(OUTPUT_DIR)urban_2016_rgb.tif: urban_2016_min.tif urban_col.txt
$(PREDICTION_DIR)urban_2013_min.tif: $(URBAN_2013) urban_2016_min.tif
	source src/utils/min_calc.sh $^ $@
$(OUTPUT_DIR)urban_2013_rgb.tif: urban_2013_min.tif urban_col.txt


# Sorry
2013_VRT=conus_hls_median_2013.vrt
2016_VRT=conus_hls_median_2016.vrt
2020_VRT=conus_hls_median_2020.vrt

$(AG_2013):
	python src/model/run_prediction.py -p $(AG_PREFIX)_2013 -r $(AG_RUNID) -f $(2013_VRT)

$(AG_2016):
	python src/model/run_prediction.py -p $(AG_PREFIX)_2016 -r $(AG_RUNID) -f $(2016_VRT)

$(AG_2020):
	python src/model/run_prediction.py -p $(AG_PREFIX)_2020 -r $(AG_RUNID) -f $(2020_VRT)

$(TRANS_2013):
	python src/model/run_prediction.py -p $(TRANS_PREFIX)_2013 -r $(TRANS_RUNID) -f $(2013_VRT)

$(TRANS_2016):
	python src/model/run_prediction.py -p $(TRANS_PREFIX)_2016 -r $(TRANS_RUNID) -f $(2016_VRT)

$(TRANS_2020):
	python src/model/run_prediction.py -p $(TRANS_PREFIX)_2020 -r $(TRANS_RUNID) -f $(2020_VRT)

$(URBAN_2013):
	python src/model/run_prediction.py -p $(URBAN_PREFIX)_2013 -r $(URBAN_RUNID) -f $(2013_VRT)

$(URBAN_2016):
	python src/model/run_prediction.py -p $(URBAN_PREFIX)_2016 -r $(URBAN_RUNID) -f $(2016_VRT)

$(URBAN_2020):
	python src/model/run_prediction.py -p $(URBAN_PREFIX)_2020 -r $(URBAN_RUNID) -f $(2020_VRT)
