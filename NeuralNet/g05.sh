#!/bin/bash

echo '25000 word vocabulary, 0.75 capture fraction'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 25000 \
         --capturefraction 0.75 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs \
         --gpuid 4
