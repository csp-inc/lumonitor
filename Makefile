SHELL=/usr/bin/env bash
PREDICTION_DIR=data/predictions/
OUTPUT_DIR=data/output/
VPATH=data/:$(PREDICTION_DIR):$(OUTPUT_DIR)


AG_2013=$(PREDICTION_DIR)hag_s_2013_hm-2016_1637276166_2d4f1735/hag_s_prediction_conus.tif
AG_2016=$(PREDICTION_DIR)hag_s_2016_hm-2016_1637276166_2d4f1735/hag_s_2016_prediction_conus.tif
AG_2020=$(PREDICTION_DIR)hag_s_2020_hm-2016_1637276166_2d4f1735/hag_s_2020_prediction_conus.tif
TRANS_2013=$(PREDICTION_DIR)htrans_2013_hm-2016_1636478645_6b6a8ce5/htrans_conus_prediction.tif
TRANS_2016=$(PREDICTION_DIR)htrans_2016_hm-2016_1636478645_6b6a8ce5/htrans_2016_prediction_conus.tif
TRANS_2020=$(PREDICTION_DIR)htrans_2020_hm-2016_1636478645_6b6a8ce5/htrans_2020_prediction_conus.tif
URBAN_2013=$(PREDICTION_DIR)urban_c_2013_hm-2016_1637623568_31c2a6cd/urban_c_2013_prediction_conus.tif
URBAN_2016=$(PREDICTION_DIR)
URBAN_2020=$(PREDICTION_DIR)

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
$(OUTPUT_DIR)atu_2013_blend_rgb.tif: au_2013_blend_rgb.tif urban_2013_rgb.tif
$(OUTPUT_DIR)atu_2016_blend_rgb.tif: au_2016_blend_rgb.tif urban_2016_rgb.tif
$(OUTPUT_DIR)atu_2020_blend_rgb.tif: au_2020_blend_rgb.tif urban_2020_rgb.tif

# For ag, just use what the model returns
$(OUTPUT_DIR)ag_2013_rgb.tif: $(AG_2013) ag_col.txt
$(OUTPUT_DIR)ag_2016_rgb.tif: $(AG_2016) ag_col.txt
$(OUTPUT_DIR)ag_2020_rgb.tif: $(AG_2020) ag_col.txt

# For trans and urban, do max calcs for 2016 & 2020
$(OUTPUT_DIR)trans_2013_rgb.tif: $(TRANS_2013) trans_col.txt
$(OUTPUT_DIR)trans_2016_rgb.tif: trans_2016_max.tif trans_col.txt
$(PREDICTION_DIR)trans_2016_max.tif: $(TRANS_2016) $(TRANS_2013)
$(OUTPUT_DIR)trans_2020_rgb.tif: trans_2020_max.tif trans_col.txt
$(PREDICTION_DIR)trans_2020_max.tif: $(TRANS_2020) $(TRANS_2016)

$(OUTPUT_DIR)urban_2013_rgb.tif: $(TRANS_2013) urban_col.txt
$(OUTPUT_DIR)urban_2016_rgb.tif: urban_2016_max.tif urban_col.txt
$(PREDICTION_DIR)urban_2016_max.tif: $(TRANS_2016) $(TRANS_2013)
$(OUTPUT_DIR)urban_2020_rgb.tif: urban_2020_max.tif urban_col.txt
$(PREDICTION_DIR)urban_2020_max.tif: $(TRANS_2020) $(TRANS_2016)
