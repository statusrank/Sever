python3 labelFlip.py --dataset age --dim 1 --flip_type threshold --flip_ratio 15 --device cuda:0 --pretrained False --optim SGD --weight_decay 1e-6
python3 labelFlip.py --dataset age --dim 1 --flip_type threshold --flip_ratio 25 --device cuda:0 --pretrained False --optim Adam --weight_decay 1e-3
python3 labelFlip.py --dataset age --dim 1 --flip_type threshold --flip_ratio 35 --device cuda:0 --pretrained False --optim Adam --weight_decay 1e-3
