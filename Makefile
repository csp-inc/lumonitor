VPATH=src/
SHELL=/usr/bin/env bash

training.csv: hls.py
	source src/utils/run_in_eastus2.sh "python3 $< $@"
