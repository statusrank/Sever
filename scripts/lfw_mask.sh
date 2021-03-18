python3 labelFlip.py --dataset lfw --dim 10 --flip_type mask --flip_ratio 15 --device cuda:0 --pretrained True --optim Adam --weight_decay 1e-2
python3 labelFlip.py --dataset lfw --dim 10 --flip_type mask --flip_ratio 25 --device cuda:0 --pretrained True --optim Adam --weight_decay 8e-4
python3 labelFlip.py --dataset lfw --dim 10 --flip_type mask --flip_ratio 35 --device cuda:0 --pretrained False --optim Adam --weight_decay 1e-3
