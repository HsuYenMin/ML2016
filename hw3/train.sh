#!/bin/bash
python3 supervised.py $1
python3 semisupervised1.py $1 $2
