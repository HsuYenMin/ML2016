#!/bin/bash
python3 autoencoder.py $1
python3 semisupervised2.py $1 $2
