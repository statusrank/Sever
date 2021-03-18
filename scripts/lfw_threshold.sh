python3 labelFlip.py --dataset lfw --dim 10 --flip_type threshold --flip_ratio 15 --device cuda:2 --pretrained False --optim Adam --weight_decay 1e-3
python3 labelFlip.py --dataset lfw --dim 10 --flip_type threshold --flip_ratio 25 --device cuda:3 --pretrained False --optim Adam --weight_decay 1e-3
python3 labelFlip.py --dataset lfw --dim 10 --flip_type threshold --flip_ratio 35 --device cuda:7 --pretrained False --optim Adam --weight_decay 1e-3
