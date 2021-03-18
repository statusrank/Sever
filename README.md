# Sever: A Robust Meta-Algorithm for Stochastic Optimization
refer to branch Sever

The official Matlab code can be found at [https://github.com/hoonose/sever.](https://github.com/hoonose/sever)

We adopt NumPy to reproduce a **SVM classification defense** algorithm.

 **You can also use the MinPy to further accelerate this algorithm**, that is, just replace 

        "import numpy as np " By

        "import MinPy.numpy as np" !
## Details
     **For all image data, we adopt Resnet50 pretrained with imagenet to extract 1000-d features and fix and feed to the downstream svm classification defense.** 
    for age and lfw data, the epsilon is set to 0.01, while for food, set epsilon as 0.05.

## How to run
    You should run the code like the following format:
    `python3 defense.py --dataset dataname --flip_type method_name --flip_ratio int_number --optim SGD --num_round 2 --num_epoch 3 --device cuda:0` 
