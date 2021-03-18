python3 labelFlip.py --dataset age --dim 1 --flip_type mask --flip_ratio 15 --device cuda:0 --pretrained False --optim Adam --weight_decay 5e-5 
python3 labelFlip.py --dataset age --dim 1 --flip_type mask --flip_ratio 25 --device cuda:0 --pretrained False --optim Adam --weight_decay 2e-5
python3 labelFlip.py --dataset age --dim 1 --flip_type mask --flip_ratio 35 --device cuda:0 --pretrained False --optim Adam --weight_decay 2e-3
