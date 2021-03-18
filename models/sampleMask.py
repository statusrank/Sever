
import sys 
sys.path.append("..") 

from utils.logger import StreamFileLogger
import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.ageDataLoader import load_age
from utils.lfwDataLoader import load_lfw
from utils.foodDataLoader import load_food
import numpy as np
import torch.optim as optim
from torchvision.models import resnet50
import torch.nn as nn
import os

from models.resnet import Resnet50, get_loss1_loss2, get_loss1_loss2_lfw, get_loss1_loss2_food

'''
    Attacker
'''
class Attacker(Resnet50):
    
    def __init__(self, **kwargs):
        super(Attacker, self).__init__(**kwargs)
        self.weight_decay = kwargs['weight_decay']

    def train_age(self, train_loader, mask_loader):
        if self.dataset == 'age':
            self.model.train()

            train_loss = 0
            for data in zip(train_loader, mask_loader):
                self.optimizer.zero_grad()

                img1, img2, label = data[0][2].to(self.device), data[0][3].to(self.device), data[0][4].to(self.device)
                mask_batch = data[1][0].to(self.device, torch.float32)
                label = label * mask_batch

                rank_score1 = self.model(img1)
                rank_score2 = self.model(img2)
                score_diff = (rank_score1 - rank_score2).squeeze()

                loss_batch = (score_diff - label) ** 2 * 0.5

                objective = loss_batch.mean() + self.get_reg_loss()

                objective.backward()
                self.optimizer.step()

                train_loss = train_loss + objective.item()

            self.scheduler.step()
            return train_loss / len(train_loader)
        elif self.dataset == 'lfw':
            self.model.train()

            train_loss = 0
            for data in zip(train_loader, mask_loader):
                self.optimizer.zero_grad()

                img1, img2, label = data[0][2].to(self.device), data[0][3].to(self.device), data[0][4].to(self.device)
                attr_id = data[0][5].to(self.device).long().view(-1,1)

                mask_batch = data[1][0].to(self.device, torch.float32)
                label = label * mask_batch

                rank_score1 = self.model(img1).gather(1,attr_id)
                rank_score2 = self.model(img2).gather(1,attr_id)
                score_diff = (rank_score1 - rank_score2).squeeze()

                loss_batch = (score_diff - label) ** 2 * 0.5

                objective = loss_batch.mean()

                objective.backward()
                self.optimizer.step()

                train_loss = train_loss + objective.item()

            self.scheduler.step()
            return train_loss / len(train_loader)
        elif self.dataset == 'food':
            self.model.train()

            train_loss = 0
            for data in zip(train_loader, mask_loader):
                self.optimizer.zero_grad()

                mask_batch = data[1][0].to(self.device, torch.int32)

                anchor = data[0][3].to(self.device).clone()
                pos = data[0][4].to(self.device).clone()
                neg = data[0][5].to(self.device).clone()

                flips = mask_batch==-1
                temp = pos[flips].clone()
                pos[flips]=neg[flips].clone()
                neg[flips]=temp

                output1 = self.model(anchor)
                output2 = self.model(pos)
                output3 = self.model(neg)
                
                triplet_loss = self.loss_fn(output1, output2, output3)

                triplet_loss.backward()
                self.optimizer.step()
                
                train_loss += triplet_loss.item()
            self.scheduler.step()
            return train_loss / len(train_loader)


'''
    utils
'''
def np_arr_2_data_loader(np_arr, batch_size):
    tensor = torch.from_numpy(np_arr)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)

# update mask
def update_mask(l1, l2, g1, g2, flip_num, data_size):
    delta = l2 - g2 - (l1-g1)
    arg_sort = np.argsort(delta)
    mask = np.ones(data_size)
    mask[arg_sort[:flip_num]] = -1

    return mask

def get_g1_g2(dataset, dim, device, img_loader, data_loader_wo_img):
    clean_path = 'checkpt/{}/clean_0.pkl'.format(dataset)
    if not os.path.exists(clean_path):
        print('Error: defender path({}) doesn\'t exist!'.format(clean_path))
        exit()

    defender_model = resnet50(pretrained=False)
    defender_model.fc = nn.Linear(2048, dim)
    defender_model.to(device)

    defender_model.load_state_dict(torch.load(clean_path, map_location=device)['model'])

    if dataset == 'age':
        return get_loss1_loss2(defender_model, img_loader, device, data_loader_wo_img)
    elif dataset == 'lfw':
        return get_loss1_loss2_lfw(defender_model, img_loader, device, data_loader_wo_img)
    elif dataset == 'food':
        return get_loss1_loss2_food(defender_model, img_loader, device, data_loader_wo_img)

def get_outer_loss(l1, l2, g1, g2, flip_num, data_size, model, weight_decay):
    part1 = np.mean(l1 - g1)

    delta = l2 - g2 - (l1-g1)
    sorted_arr = np.sort(delta)

    part2 = np.sum(sorted_arr[:flip_num]) / data_size

    reg_loss = 0
    for param in model.parameters():
        reg_loss += (param.data ** 2).sum()

    return part1 + part2 + weight_decay * reg_loss.item()

'''
    mask flip
'''
load_data = {
    'age': load_age,
    'lfw': load_lfw,
    'food': load_food
}

def mask_flip(**kwargs):
    # logger
    logger = StreamFileLogger(kwargs['name'], kwargs['log_dir'])

    mask = np.ones(kwargs['data_size'])

    print('==> loading data ...')
    _, train_loader, _, img_loader, _, train_loader_wo_img = load_data[kwargs['attacker_params']['dataset']](**kwargs['data_params'])

    print('==> calculating g1 and g2')
    g1, g2 = get_g1_g2(kwargs['attacker_params']['dataset'], 
                    kwargs['attacker_params']['dim'], kwargs['attacker_params']['device'], 
                    img_loader, train_loader_wo_img)

    print('==> creating attacker ...')
    attacker = Attacker(**kwargs['attacker_params'])

    for outer in range(kwargs['outer']):
        logger.record('==> updating W ...')
        mask_loader = np_arr_2_data_loader(mask, kwargs['data_params']['batch_size'])

        for inner in range(kwargs['inner']):
            inner_loss = attacker.train_age(train_loader, mask_loader)

            l1, l2 = attacker.get_loss1_loss2(img_loader, train_loader_wo_img)
            part1, part2, R = get_outer_loss(l1, l2, g1, g2, kwargs['flip_num'],
                                kwargs['data_size'], attacker,
                                kwargs['attacker_params']['weight_decay'])
            
            logger.record('inner epoch: {}; inner loss: {:.6f}; R:{:.6f}'.format(inner, inner_loss, R))


        logger.record('==> updating lambda ...')
        l1, l2 = attacker.get_loss1_loss2(img_loader, train_loader_wo_img)

        previous_mask = mask
        mask = update_mask(l1, l2, g1, g2, kwargs['flip_num'], kwargs['data_size'])
        
        outer_loss = get_outer_loss(l1, l2, g1, g2, kwargs['flip_num'],
                                kwargs['data_size'], attacker.model,
                                kwargs['attacker_params']['weight_decay'])

        mask_difference = np.sum(mask!=previous_mask)
        logger.record('outer epoch: {}; outer loss: {:.6f}; R: {:.6f}; mask difference: {}'.format(outer, outer_loss, R, mask_difference))

        if mask_difference == 0:
            break
    
    attacker.save(kwargs['outer']*kwargs['inner'], kwargs['checkpt_dir'], kwargs['name'])
    return mask