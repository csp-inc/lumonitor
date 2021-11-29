SHELL=/usr/bin/env bash
PREDICTION_DIR=data/predictions/
OUTPUT_DIR=data/output/
VPATH=data/:$(PREDICTION_DIR):$(OUTPUT_DIR)


AG_PREFIX=hag_s
AG_RUNID=hm-2016_1637276166_2d4f1735
AG_2013=$(PREDICTION_DIR)$(AG_PREFIX)_2013_$(AG_RUNID)/$(AG_PREFIX)_2013_prediction_conus.tif
AG_2016=$(PREDICTION_DIR)$(AG_PREFIX)_2016_$(AG_RUNID)/$(AG_PREFIX)_2016_prediction_conus.tif
AG_2020=$(PREDICTION_DIR)$(AG_PREFIX)_2020_$(AG_RUNID)/$(AG_PREFIX)_2020_prediction_conus.tif

TRANS_PREFIX=htrans
TRANS_RUNID=hm-2016_1636478645_6b6a8ce5
TRANS_2013=$(PREDICTION_DIR)$(TRANS_PREFIX)_2013_$(TRANS_RUNID)/$(TRANS_PREFIX)_2013_prediction_conus.tif
TRANS_2016=$(PREDICTION_DIR)$(TRANS_PREFIX)_2016_$(TRANS_RUNID)/$(TRANS_PREFIX)_2016_prediction_conus.tif
TRANS_2020=$(PREDICTION_DIR)$(TRANS_PREFIX)_2020_$(TRANS_RUNID)/$(TRANS_PREFIX)_2020_prediction_conus.tif

URBAN_PREFIX=hurban_c
URBAN_RUNID=hm-2016_1637623568_31c2a6cd
URBAN_2013=$(PREDICTION_DIR)$(URBAN_PREFIX)_2013_$(URBAN_RUNID)/$(URBAN_PREFIX)_2013_prediction_conus.tif
URBAN_2016=$(PREDICTION_DIR)$(URBAN_PREFIX)_2016_$(URBAN_RUNID)/$(URBAN_PREFIX)_2016_prediction_conus.tif
URBAN_2020=$(PREDICTION_DIR)$(URBAN_PREFIX)_2020_$(URBAN_RUNID)/$(URBAN_PREFIX)_2020_prediction_conus.tif

LAYERS=ag trans urban
BLAYERS=at tu au atu
EXT=tif

YEARS=2013 2016 2020
LAYER_YEARS=$(foreach LAYER, ${LAYERS}, $(foreach YEAR, ${YEARS}, ${LAYER}_${YEAR}))
BLAYER_YEARS=$(foreach BLAYER, ${BLAYERS}, $(foreach YEAR, ${YEARS}, ${BLAYER}_${YEAR}_blend))
RGBS=$(patsubst %, $(OUTPUT_DIR)%_rgb.$(EXT), $(LAYER_YEARS) $(BLAYER_YEARS))
TILES=$(patsubst %, %.tiles, $(LAYER_YEARS) $(BLAYER_YEARS))

tiles: $(TILES)
rgbs: $(RGBS)

%_blend_rgb.$(EXT):
	source src/utils/blend_images.sh $^ $@

%_rgb.$(EXT):
	source src/utils/export_rgb.sh $^ $@

%.tiles: %_rgb.$(EXT)
	source src/utils/make_raster_tiles.sh $^
	touch data/$@

# NOt the prettiest, I admit
$(OUTPUT_DIR)at_2013_blend_rgb.$(EXT): ag_2013_rgb.$(EXT) trans_2013_rgb.$(EXT)
$(OUTPUT_DIR)at_2016_blend_rgb.$(EXT): ag_2016_rgb.$(EXT) trans_2016_rgb.$(EXT)
$(OUTPUT_DIR)at_2020_blend_rgb.$(EXT): ag_2020_rgb.$(EXT) trans_2020_rgb.$(EXT)
$(OUTPUT_DIR)au_2013_blend_rgb.$(EXT): ag_2013_rgb.$(EXT) urban_2013_rgb.$(EXT)
$(OUTPUT_DIR)au_2016_blend_rgb.$(EXT): ag_2016_rgb.$(EXT) urban_2016_rgb.$(EXT)
$(OUTPUT_DIR)au_2020_blend_rgb.$(EXT): ag_2020_rgb.$(EXT) urban_2020_rgb.$(EXT)
$(OUTPUT_DIR)tu_2013_blend_rgb.$(EXT): trans_2013_rgb.$(EXT) urban_2013_rgb.$(EXT)
$(OUTPUT_DIR)tu_2016_blend_rgb.$(EXT): trans_2016_rgb.$(EXT) urban_2016_rgb.$(EXT)
$(OUTPUT_DIR)tu_2020_blend_rgb.$(EXT): trans_2020_rgb.$(EXT) urban_2020_rgb.$(EXT)
$(OUTPUT_DIR)atu_2013_blend_rgb.$(EXT): tu_2013_blend_rgb.$(EXT) ag_2013_rgb.$(EXT)
$(OUTPUT_DIR)atu_2016_blend_rgb.$(EXT): tu_2016_blend_rgb.$(EXT) ag_2016_rgb.$(EXT)
$(OUTPUT_DIR)atu_2020_blend_rgb.$(EXT): tu_2020_blend_rgb.$(EXT) ag_2020_rgb.$(EXT)

# For ag, just use what the model returns
$(OUTPUT_DIR)ag_2013_rgb.$(EXT): $(AG_2013) ag_col.txt
$(OUTPUT_DIR)ag_2016_rgb.$(EXT): $(AG_2016) ag_col.txt
$(OUTPUT_DIR)ag_2020_rgb.$(EXT): $(AG_2020) ag_col.txt

# For trans and urban, do max calcs for 2016 & 2020
$(OUTPUT_DIR)trans_2013_rgb.$(EXT): $(TRANS_2013) trans_col.txt
$(OUTPUT_DIR)trans_2016_rgb.$(EXT): trans_2016_max.$(EXT) trans_col.txt
$(PREDICTION_DIR)trans_2016_max.$(EXT): $(TRANS_2016) $(TRANS_2013)
	source src/utils/max_calc.sh $^ $@
$(OUTPUT_DIR)trans_2020_rgb.$(EXT): trans_2020_max.$(EXT) trans_col.txt
$(PREDICTION_DIR)trans_2020_max.$(EXT): $(TRANS_2020) $(TRANS_2016)
	source src/utils/max_calc.sh $^ $@

$(OUTPUT_DIR)urban_2013_rgb.$(EXT): $(URBAN_2013) urban_col.txt
$(OUTPUT_DIR)urban_2016_rgb.$(EXT): urban_2016_max.$(EXT) urban_col.txt
$(PREDICTION_DIR)urban_2016_max.$(EXT): $(URBAN_2016) $(URBAN_2013)
	source src/utils/max_calc.sh $^ $@
$(OUTPUT_DIR)urban_2020_rgb.$(EXT): urban_2020_max.$(EXT) urban_col.txt
$(PREDICTION_DIR)urban_2020_max.$(EXT): $(URBAN_2020) $(URBAN_2016)
	source src/utils/max_calc.sh $^ $@

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
