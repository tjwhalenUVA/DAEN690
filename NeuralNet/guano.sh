#!/bin/bash

echo '100 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 100 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '200 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 200 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '500 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 500 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '1000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 1000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '2000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 2000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '5000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 5000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '10000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 10000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '20000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 20000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '25000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 25000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs

echo '50000 word vocabulary'
/home/DAEN690/gitHub/NeuralNet/CNN.py --inputfile /home/DAEN690/gitHub/data/articles_zenodo.db \
         --glovefile /home/DAEN690/gitHub/data/glove.6B.50d.txt \
         --vocabsize 50000 \
         --capturefraction 0.95 \
         --crossvalidate \
         --folds 10 \
         --epochs 10 \
         --convolutionfilters 50 \
         --kernel 5 \
         --poollayer 5 \
         --flattenlayer \
         --denselayer 50 \
         --verbose 2 \
         --graphs
