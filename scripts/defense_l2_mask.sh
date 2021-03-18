# python3 defense.py --dataset age --flip_type l2_mask --flip_ratio 15 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset age --flip_type l2_mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset age --flip_type l2_mask --flip_ratio 35 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0



# python3 defense.py --dataset lfw --flip_type l2_mask --flip_ratio 15 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type l2_mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type l2_mask --flip_ratio 35 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0



# python3 defense.py --dataset food --flip_type l2_mask --flip_ratio 15 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset food --flip_type l2_mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset food --flip_type l2_mask --flip_ratio 35 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0


python3 defense.py --dataset food --flip_type pga --flip_ratio 15 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
python3 defense.py --dataset food --flip_type pga --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
python3 defense.py --dataset food --flip_type pga --flip_ratio 35 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0