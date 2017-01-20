#!/bin/bash

python3 csv2vw.py
vw -d train.w -f nn_250.vw -c --loss_function logistic -l 0.01 -b 28 --l2 1e-7 --l1 1e-5 --ftrl --ignore b --nn 250 --dropout --passes 2
vw test.w -t -i nn_250.vw -p prob_nn_250.txt
Rscript write_submission.R