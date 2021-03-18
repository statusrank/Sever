# python3 defense.py --dataset age --flip_type threshold --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset age --flip_type mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset age --flip_type furthest --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset age --flip_type nearest --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
python3 defense.py --dataset age --flip_type reverse --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
python3 defense.py --dataset age --flip_type random --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0

python3 defense.py --dataset age --flip_type pga --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
python3 defense.py --dataset age --flip_type l2_mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0



# python3 defense.py --dataset lfw --flip_type threshold --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type furthest --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type nearest --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type reverse --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type random --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0

# python3 defense.py --dataset lfw --flip_type pga --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0
# python3 defense.py --dataset lfw --flip_type l2_mask --flip_ratio 25 --optim SGD --num_round 2 --num_epoch 3 --device cuda:0