
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

from models.resnet import Resnet50, get_loss1_loss2
from models.sampleMask import update_mask, np_arr_2_data_loader, get_g1_g2, get_loss1_loss2, get_outer_loss

'''
    Attacker
'''
class Attacker(Resnet50):
    
    def __init__(self, **kwargs):
        super(Attacker, self).__init__(**kwargs)
        
        if self.dataset == 'food':
            self.loss_fn.reduction = 'none'

        multisteps = [8, 14]
        self.lamb = torch.tensor(0., requires_grad=True, device=self.device)
        self.optimizer_lamb = optim.Adam([self.lamb], lr=1e-5)
        self.scheduler_lamb = optim.lr_scheduler.MultiStepLR(
            self.optimizer_lamb, milestones=multisteps, gamma=kwargs['gamma'])


    def train_age(self, train_loader, g1_loader, g2_loader):
        if self.dataset == 'age':
            self.model.train()

            train_loss = 0
            for data in zip(train_loader, g1_loader, g2_loader):
                self.optimizer.zero_grad()

                img1, img2, label = data[0][2].to(self.device), data[0][3].to(self.device), data[0][4].to(self.device)
                g1_batch = data[1][0].to(self.device, torch.float32)
                g2_batch = data[2][0].to(self.device, torch.float32)

                rank_score1 = self.model(img1)
                rank_score2 = self.model(img2)
                score_diff = (rank_score1 - rank_score2).squeeze()

                l1_batch = (score_diff - label) ** 2 * 0.5
                l2_batch = (score_diff + label) ** 2 * 0.5

                delta = l2_batch - g2_batch - (l1_batch-g1_batch)

                part1 = (l1_batch-g1_batch).mean()
                part2 = torch.clamp(-delta-self.lamb, min = 0).mean()
                objective = part1 - part2

                objective.backward()

                self.optimizer.step()

                train_loss = train_loss + objective.item()

            self.scheduler.step()
            return train_loss / len(train_loader)
        elif self.dataset == 'lfw':
            self.model.train()

            train_loss = 0
            for data in zip(train_loader, g1_loader, g2_loader):
                self.optimizer.zero_grad()

                img1, img2, label = data[0][2].to(self.device), data[0][3].to(self.device), data[0][4].to(self.device)
                attr_id = data[0][5].to(self.device).long().view(-1,1)

                g1_batch = data[1][0].to(self.device, torch.float32)
                g2_batch = data[2][0].to(self.device, torch.float32)

                rank_score1 = self.model(img1).gather(1,attr_id)
                rank_score2 = self.model(img2).gather(1,attr_id)
                score_diff = (rank_score1 - rank_score2).squeeze()

                l1_batch = (score_diff - label) ** 2 * 0.5
                l2_batch = (score_diff + label) ** 2 * 0.5

                delta = l2_batch - g2_batch - (l1_batch-g1_batch)

                part1 = (l1_batch-g1_batch).mean()
                part2 = torch.clamp(-delta-self.lamb, min = 0).mean()
                objective = part1 - part2

                objective.backward()

                self.optimizer.step()

                train_loss = train_loss + objective.item()

            self.scheduler.step()
            return train_loss / len(train_loader)
        elif self.dataset == 'food':
            self.model.train()

            train_loss = 0
            for data in zip(train_loader, g1_loader, g2_loader):
                self.optimizer.zero_grad()

                anchor, pos, neg = data[0][3].to(self.device), data[0][4].to(self.device), data[0][5].to(self.device)

                g1_batch = data[1][0].to(self.device, torch.float32)
                g2_batch = data[2][0].to(self.device, torch.float32)

                output1 = self.model(anchor)
                output2 = self.model(pos)
                output3 = self.model(neg)

                l1_batch = self.loss_fn(output1, output2, output3)
                l2_batch = self.loss_fn(output1, output3, output2)

                delta = l2_batch - g2_batch - (l1_batch-g1_batch)

                part1 = (l1_batch-g1_batch).mean()
                part2 = torch.clamp(-delta-self.lamb, min = 0).mean()
                objective = part1 - part2

                objective.backward()

                self.optimizer.step()

                train_loss = train_loss + objective.item()

            self.scheduler.step()
            return train_loss / len(train_loader)

    
    def update_lambda(self, l1_loader, l2_loader, g1_loader, g2_loader, flip_num, data_size):
        self.model.eval()

        train_loss = 0
        for data in zip(l1_loader, l2_loader, g1_loader, g2_loader):
            self.optimizer_lamb.zero_grad()

            l1_batch = data[0][0].to(self.device, torch.float32)
            l2_batch = data[1][0].to(self.device, torch.float32)
            g1_batch = data[2][0].to(self.device, torch.float32)
            g2_batch = data[3][0].to(self.device, torch.float32)

            delta = l2_batch - g2_batch - (l1_batch-g1_batch)

            bz = l1_batch.shape[0]
            objective = flip_num * self.lamb / data_size * bz + torch.sum(torch.clamp(-delta-self.lamb, min = 0))
            objective = objective / bz 

            objective.backward()
            nn.utils.clip_grad_norm_([self.lamb], 1e5, norm_type=1) 
            
            self.optimizer_lamb.step()

            train_loss = train_loss + objective.item() 

        self.scheduler_lamb.step()
        return train_loss / len(l1_loader)

    def save(self, epoch, checkpt, name):
        path = os.path.join(checkpt, name+'.pkl')
        torch.save({'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'lambda': self.lamb,
                    'optimizer_lamb': self.optimizer_lamb.state_dict(),
                    'scheduler_lamb': self.scheduler_lamb.state_dict()},
                    path)

'''
    threshold flip
'''
load_data = {
    'age': load_age,
    'lfw': load_lfw,
    'food': load_food
}
def threshold_flip(**kwargs):
    # logger
    logger = StreamFileLogger(kwargs['name'], kwargs['log_dir'])

    print('==> loading data ...')
    _, train_loader, _, img_loader, _, train_loader_wo_img = load_data[kwargs['attacker_params']['dataset']](**kwargs['data_params'])

    print('==> calculating g1 and g2')
    g1, g2 = get_g1_g2(kwargs['attacker_params']['dataset'], 
                    kwargs['attacker_params']['dim'], kwargs['attacker_params']['device'], 
                    img_loader, train_loader_wo_img)
    g1_loader = np_arr_2_data_loader(g1, kwargs['data_params']['batch_size'])
    g2_loader = np_arr_2_data_loader(g2, kwargs['data_params']['batch_size'])

    print('==> creating attacker ...')
    attacker = Attacker(**kwargs['attacker_params'])

    mask = np.ones(kwargs['data_size'])

    for outer in range(kwargs['outer']):
        logger.record('==> updating W ...')
        
        for inner in range(kwargs['inner']):
            inner_loss = attacker.train_age(train_loader, g1_loader, g2_loader)
            logger.record('inner epoch: {}; inner loss: {:.6f}'.format(inner, inner_loss))

        logger.record('==> updating lambda ...')
        l1, l2 = attacker.get_loss1_loss2(img_loader, train_loader_wo_img)
        l1_loader = np_arr_2_data_loader(l1, kwargs['data_params']['batch_size'])
        l2_loader = np_arr_2_data_loader(l2, kwargs['data_params']['batch_size'])


        for _ in range(2):
            attacker.update_lambda(l1_loader, l2_loader, 
                        g1_loader, g2_loader, 
                        kwargs['flip_num'], kwargs['data_size'])
        
        outer_loss = get_outer_loss(l1, l2, g1, g2, kwargs['flip_num'],
                                kwargs['data_size'], attacker.model,
                                kwargs['attacker_params']['weight_decay'])
        previous_mask = mask
        mask = update_mask(l1, l2, g1, g2, kwargs['flip_num'], kwargs['data_size'])
        
        mask_difference = np.sum(mask!=previous_mask)
        logger.record('outer epoch: {}; outer loss: {:.6f}; mask difference: {}'.format(outer, outer_loss, mask_difference))

        if mask_difference == 0:
            break

    attacker.save(kwargs['outer']*kwargs['inner'], kwargs['checkpt_dir'], kwargs['name'])

    return update_mask(l1, l2, g1, g2, kwargs['flip_num'], kwargs['data_size'])