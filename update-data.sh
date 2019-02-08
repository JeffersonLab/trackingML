#!/bin/bash

DATA_URL=http://clasweb.jlab.org/ML/data

HALL_B_DATA=traking_data_hallb.targ.gz
HALL_D_DATA=traking_data_halld.targ.gz
HALL_E_DATA=traking_data_halle.targ.gz

wget $DATA_URL/$HALL_B_DATA
mv $HALL_B_DATA data/training/hall-b/
mv $HALL_D_DATA data/training/hall-d/
mv $HALL_E_DATA data/training/eic/
