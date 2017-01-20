#!/bin/bash

python3 leak.py
python3 data.py
./ffm-train -s 4 --on-disk --no-rand data/train model
./ffm-predict data/test model output
python3 data2.py