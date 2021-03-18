# Rethinking Label Flipping Attack: From Sample Masking to Sample Thresholding

This is a PyTorch implementation of the experiments in our paper:
> xxxxxxx

# Prerequisites
* Linux or macOS
* Python 3
* NVIDIA GPU + CUDA CuDNN
* R + ggplot2 (for data visualization)

# Dependencies
* PyTorch >= 1.4.0
* torchvision >= 0.2.2
* NumPy

# Getting Started
## Installation
* Clone this repo:  
  ```xxxxxx```
* Install all the aforementioned dependencies.
## Prepare Data
* Unzip  ` ./data/{dataset}/imgs.zip ` . Here is an example:  
  ``` unzip ./data/age/imgs.zip -d ./data/age/imgs ```
## Pre-train Clean Victim Model
* Here is an example to pre-train clean victim model on Human Age Dataset.  
  ```python3 main.py --dataset age --dim 1 --flip_type clean --flip_ratio 0 --pretrained False --device cuda:0```
* Both your screen and a log file (`  ./log/{dataset}/clean_0.log  `) will simultaneously record loss and metrics for every epoch.
* Once it runs out, it will output a clean victim model (`  ./checkpt/{dataset}/clean_0.pkl  `).
* Please change relevant arguments for different datasets: 
  - LFW-10: `--dataset lfw --dim 10 --pretrained True`
  - Food: `--dataset food --dim 5`

## Conduct Label Flip Attack
* After pre-training a clean victim model, 
  ```python3 labelFlip.py --dataset age --dim 1 --flip_type threshold --flip_ratio 15 --device cuda:7 --pretrained False --optim SGD --weight_decay 1e-6```
* For more scripts, please see `  ./scripts/{dataset}_{flip type}.sh  `.
* For mask ( Sample Masking ) and threshold ( Sample Thresholding ), intermediate results will be recorded into a log file (`  ./log/{dataset}/attacker/{flip type}_{flip ratio}.log  `). Besides, the final attcker model will be saved at ` ./checkpt/{dataset}/{flip type}_{flip ratio}.pkl `.
* Corrupted training set will be saved at ` ./data/{dataset}/train/{flip type}_{flip ratio}.txt `

## Train/Test Corrupt Victim Model
* Similar to the pre-training of clean victim model, we train/test corrupt victim models by selecting different flip type and flip ratio.  
  ```python3 main.py --dataset age --dim 1 --flip_type threshold --flip_ratio 15 --pretrained False --device cuda:0```
* Both your screen and a log file (`  ./log/{dataset}/{flip type}_{flip ratio}.log  `) will simultaneously record loss and metrics for every epoch.
* Once it runs out, it will output a corrupt victim model (`  ./checkpt/{dataset}/{flip type}_{flip ratio}.pkl  `).
* Please change relevant arguments for different settings.

# Visualization
* Change the current working directory to `./plot` .  
  ``` cd ./plot ```
## Density Plot of Incorrectly Predicted Pairs on Human Age Dataset Incorrectly Predicted Pairs
* To get the density plot, we first need to figure out all incorrectly predicted pairs and their respective score differences, by running ` python3 ./getAgeDensity.py ` and generating  ` ./age/density_0.txt ` .
* Turn to R and run ` ./Rscripts/densityPlot.R ` .
## Histogram & Scatter of Score Difference of Flipped Pairs on Training Data 
* Here is an example to obtain [index, score difference, flip type] of flipped pairs by running ` python3 ./getFlip.py --dataset age --dim 1 --device cuda:0` . Such command will output 3 files which are ` ./age/flip_{15, 25, 35}.txt ` .
* Turn to R. Run ` ./Rscripts/flipHistPlot.R ` and ` ./Rscripts/flipScatterPlot.R ` to get histogram and scatter plot separately based on the aforementioned txt file.
##  Histograms of Score Difference of Successfully Attacked Cases on Test Set
* Here is an example to obtain [score difference, flip type] of wrongly successfully attacked cases on test set by running ` python3 ./getPred.py --dataset age --dim 1 --device cuda:0` . Such command will output 3 files which are ` ./age/pred_{15, 25, 35}.txt ` .
* Turn to R. Run ` ./Rscripts/predHistPlot.R ` to get the histogram based on the aforementioned txt file.
  
# Code Structure
To help users better understand and use our codebase, we briefly overview the functionality and implementation of each package and each module.

main.py is a general-purpose training and testing script. It works for various datasets (with option `--dataset` and option `--dim`: e.g., [`age, 1`], [`lfw, 10`], [`food, 5`]), and different attacks (with option `--flip_type`: e.g. `clean`, `nearest`, `random`, `reverse`, `furthest`, `mask`, `threshold`, and `-- flip_ratio`: e.g. `0`, `15`, `25`, `35`).

labelFlip.py is an entrance for all the label-flipping methods. It will generate the corrupt training set according to option `--dataset`, `--flip_type` and `--flip_ratio`. It also contains implementations of 4 attacks (i.e. Nearest, Random, Reverse and Furthest)

data directory contains all data-related items.
* {dataset}
  * train contains all training sets including corrupt ones.
  * test.txt
  * imgs

pretrained/age_estimation_resnet50.pth.tar is used during the training process of Human Age dataset.

models directory contains modules related to objective functions, optimizations, and network architectures.
* resnet.py implements the Siamese Ranknet for Human Age dataset and LFW-10 dataset, and the Triplet Network for Food dataset.
* sampleMask.py implements the Sample Masking algorithm, including defining its `Attacker`, inheriting from `Resnet50` defined in resnet.py.
* sampleThreshold.py implements the Sample Threshold algorithm, including defining its `Attacker`, which inherits from `Resnet50` defined in resnet.py.
  
utils directory includes a miscellaneous collection of useful helper functions.
* {dataset}DataLoader.py is responsible for data loading and preprocessing.
* logger.py includes a class named `StreamFileLogger` for printing/saving logging information.
# Citation
Please cite our paper if you use this code in your own work:  
 ``` xxxxxxx ```
