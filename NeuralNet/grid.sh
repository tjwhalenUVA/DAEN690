#!/bin/bash

echo "Performing grid search using $1 as the hyperparameter file"

./NeuralNet/CNNgrid.py --inputfile ./data/articles_zenodo.db \
         --glovefile ./data/glove.6B.50d.txt \
         --vocabsize 25000 \
         --capturefraction 0.95 \
         --gridsearch $1 \
         --epochs 2 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 1 \
         --graphs

