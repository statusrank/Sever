import torch
import numpy as np
import torch.optim as optim
from torchvision.models import resnet50
import torch.nn as nn
from torch.nn.functional import triplet_margin_loss
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import os

class Resnet50():
    def __weight_init(self):
        if isinstance(self.model.fc, nn.Linear):
            nn.init.normal_(self.model.fc.weight.data, 0, 0.01)
            nn.init.constant_(self.model.fc.bias.data, 0)

    def __init__(self, **kwargs):
        self.device = kwargs['device']
        if self.device.startswith('cuda'):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        self.model = resnet50(pretrained=kwargs['pretrained'])
        self.dataset = kwargs['dataset']
        if self.dataset == 'age':
            self.model.fc = nn.Linear(2048, 62)
            
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            state_dict = torch.load('pretrained/age_estimation_resnet50.pth.tar', 
                                    map_location=(self.device))['state_dict']
            for k, v in state_dict.items():
                if 'num_batches_tracked' not in k:
                    new_state_dict[k[7:]] = v

            resnet50_dict = self.model.state_dict()
            resnet50_dict.update(new_state_dict)
            self.model.load_state_dict(resnet50_dict)

        self.dim = kwargs['dim']
        self.model.fc = nn.Linear(2048, self.dim)
        self.__weight_init()
        self.model.to(self.device)

        if self.dataset == 'food':
            self.loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)

        if kwargs['optim'] == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(), 
                lr=kwargs['lr'], momentum=kwargs['momentum'],
                weight_decay=kwargs['weight_decay']
            )
        elif kwargs['optim'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), 
                lr=kwargs['lr'],
                weight_decay=kwargs['weight_decay']
            )
        
        multisteps = [40, 60, 75, 85, 90]
        self.scheduler = optim.lr_scheduler.MultiStepLR(
                        self.optimizer, milestones=multisteps, gamma=kwargs['gamma'])

    
    def train_age(self, train_loader):
        self.model.train()

        train_loss = 0 
        correct = 0

        for data in train_loader:
            self.optimizer.zero_grad()
    
            img1, img2, label = data[2].to(self.device), data[3].to(self.device), data[4].to(self.device)

            rank_score1 = self.model(img1)
            rank_score2 = self.model(img2)
            score_diff = (rank_score1 - rank_score2).squeeze()

            loss_batch = (score_diff - label) ** 2 * 0.5

            loss = loss_batch.mean()

            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()

            predictions = -torch.ones_like(score_diff)
            predictions[score_diff>0] = 1

            correct = correct + torch.sum(predictions==label)
        
        self.scheduler.step()
        return train_loss / len(train_loader), float(correct) / len(train_loader.dataset)

    def get_all_rank_scores(self, img_loader):
        self.model.eval()

        score_dict = {}
        with torch.no_grad():
            for img, img_id, _ in img_loader:
                img_id, img = img_id.to(self.device), img.to(self.device)
                output = self.model(img)

                for i, o in zip(img_id, output):
                    score_dict[int(i)] = o.item()
        
        return score_dict

    def get_loss1_loss2(self, img_loader, data_loader_wo_img):
        self.model.eval()

        loss1 = []
        loss2 = []

        with torch.no_grad():
            if self.dataset == 'age':
                score_dict = self.get_all_rank_scores(img_loader)

                for img_id1, img_id2, label, _, _, _ in data_loader_wo_img:
                    img_id1, img_id2 = img_id1.to(self.device), img_id2.to(self.device)
                    label = label.to(self.device)

                    for i1, i2, l in zip(img_id1, img_id2, label):
                        score_diff = score_dict[int(i1)] - score_dict[int(i2)]
                        loss1.append((score_diff - l) ** 2 * 0.5)
                        loss2.append((score_diff + l) ** 2 * 0.5)
            elif self.dataset == 'lfw':
                score_dict = self.get_all_rank_scores_lfw(img_loader)

                for img_id1, img_id2, label, attr_id, _, _ in data_loader_wo_img:
                    img_id1, img_id2 = img_id1.to(self.device), img_id2.to(self.device)
                    label, attr_id = label.to(self.device), attr_id.to(self.device)

                    for i1, i2, l, a in zip(img_id1, img_id2, label, attr_id):
                        score_diff = score_dict[int(i1)][int(a)].item() - score_dict[int(i2)][int(a)].item()
                        loss1.append((score_diff - l) ** 2 * 0.5)
                        loss2.append((score_diff + l) ** 2 * 0.5)
            elif self.dataset == 'food':
                score_dict = self.get_all_rank_scores_lfw(img_loader)

                for anc, pos, neg, _ in data_loader_wo_img:
                    anc, pos, neg = anc.to(self.device), pos.to(self.device), neg.to(self.device)

                    for a, p, n in zip(anc, pos, neg):
                        output1 = score_dict[int(a)].view(1,-1)
                        output2 = score_dict[int(p)].view(1,-1)
                        output3 = score_dict[int(n)].view(1,-1)
                        l1 = triplet_margin_loss(output1, output2, output3, margin=1.0, p=2)
                        l2 = triplet_margin_loss(output1, output3, output2, margin=1.0, p=2)
                        
                        loss1.append(l1.item())
                        loss2.append(l2.item())


        return np.array(loss1, dtype=float), np.array(loss2, dtype=float)

    def test_age(self, img_loader, test_loader_wo_img):
        self.model.eval()

        test_loss = 0
        test_acc = 0
        f1, prec, rec, auc = 0, 0, 0, 0

        scores = self.get_all_rank_scores(img_loader)

        test_size = len(test_loader_wo_img.dataset)
        label_arr = []
        diff_arr = []
        
        with torch.no_grad():
            for data in test_loader_wo_img:
                img_id1, img_id2, label = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                for id_1, id_2, l in zip(img_id1, img_id2, label):
                    label_arr.append(int(l))
                    diff_arr.append(scores[int(id_1)] - scores[int(id_2)])

        label_arr, diff_arr = np.array(label_arr), np.array(diff_arr)

        test_loss = np.sum((diff_arr-label_arr)**2*0.5) / test_size

        predictions = -np.ones(test_size, dtype=int)
        predictions[diff_arr>0] = 1
        metrics = [accuracy_score, f1_score, precision_score, recall_score]
        test_acc, f1, prec, rec = [func(label_arr, predictions) for func in metrics]
        auc = roc_auc_score(label_arr, diff_arr)
        return test_loss, test_acc, f1, prec, rec, auc

    def train_lfw(self, train_loader):
        self.model.train()

        train_loss = 0 #TODO
        correct = 0

        for data in train_loader:
            self.optimizer.zero_grad()
    
            img1, img2, label = data[2].to(self.device), data[3].to(self.device), data[4].to(self.device)
            attr_id = data[5].to(self.device).long().view(-1,1)

            rank_score1 = self.model(img1).gather(1,attr_id)
            rank_score2 = self.model(img2).gather(1,attr_id)
            score_diff = (rank_score1 - rank_score2).squeeze()

            loss_batch = (score_diff - label) ** 2 * 0.5

            loss = loss_batch.mean()

            loss.backward()
            self.optimizer.step()

            train_loss = train_loss + loss.item()

            predictions = -torch.ones_like(score_diff)
            predictions[score_diff>0] = 1

            correct = correct + torch.sum(predictions==label)
        
        self.scheduler.step()
        return train_loss / len(train_loader), float(correct) / len(train_loader.dataset)

    def get_all_rank_scores_lfw(self, img_loader):
        self.model.eval()

        score_dict = {}
        with torch.no_grad():
            for img, img_id, _ in img_loader:
                img_id, img = img_id.to(self.device), img.to(self.device)
                output = self.model(img)

                for i, o in zip(img_id, output):
                    score_dict[int(i)] = o
        
        return score_dict

    def test_lfw(self, img_loader, test_loader_wo_img):
        self.model.eval()

        test_loss = 0
        test_acc = [[] for _ in range(self.dim)]

        scores = self.get_all_rank_scores_lfw(img_loader)

        test_size = len(test_loader_wo_img.dataset)
        label_arr = [[] for _ in range(self.dim)]
        diff_arr = [[] for _ in range(self.dim)]
        test_acc = []
        corrects = 0

        with torch.no_grad():
            for data in test_loader_wo_img:
                img_id1, img_id2, label = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)
                attr_id = data[3].to(self.device).long()

                for id_1, id_2, l, a in zip(img_id1, img_id2, label, attr_id):
                    label_arr[a].append(int(l))
                    diff_arr[a].append(scores[int(id_1)][a].item() - scores[int(id_2)][a].item())

        for l_a, d_a in zip(label_arr, diff_arr):
            l_a, d_a = np.array(l_a), np.array(d_a)
            test_loss += np.sum((l_a - d_a)**2*0.5)
            predictions = -np.ones_like(d_a, dtype=int)
            predictions[d_a>0] = 1
            corrects += np.sum(l_a==predictions)
            test_acc.append(accuracy_score(l_a, predictions))

        test_loss /= test_size

        return test_loss, corrects/test_size, test_acc

    def train_food(self, train_loader):
        self.model.train()

        train_loss = 0 
        correct = 0

        for data in train_loader:
            self.optimizer.zero_grad()

            anchor, pos, neg = (data[3].to(self.device), data[4].to(self.device),
                data[5].to(self.device))

            output1 = self.model(anchor)
            output2 = self.model(pos)
            output3 = self.model(neg)
            
            triplet_loss = self.loss_fn(output1, output2, output3)

            triplet_loss.backward()
            self.optimizer.step()
            
            train_loss += triplet_loss.item()
            dist1 = torch.norm(output1-output2, p=2, dim=1)
            dist2 = torch.norm(output1-output3, p=2, dim=1)
            correct += torch.sum(dist1<dist2)#TODO
        
        return train_loss / len(train_loader), float(correct) / len(train_loader.dataset)  
    
    def test_food(self, img_loader, test_loader_wo_img):
        self.model.eval()

        test_loss = 0
        corrects = 0

        scores = self.get_all_rank_scores_lfw(img_loader)
        with torch.no_grad():
            for data in test_loader_wo_img:
                anchor, pos, neg = data[0].to(self.device), data[1].to(self.device), data[2].to(self.device)

                for a, p, n in zip(anchor, pos, neg):
                    output1, output2, output3 = scores[int(a)], scores[int(p)], scores[int(n)]
                    dist1 = torch.norm(output1-output2, p=2).item()
                    dist2 = torch.norm(output1-output3, p=2).item()
                    
                    corrects += dist1<dist2
                    test_loss += self.loss_fn(output1.view(-1,self.dim), output2.view(-1,self.dim), output3.view(-1,self.dim))

        data_size = len(test_loader_wo_img.dataset)
        return test_loss / data_size, float(corrects) / data_size

    def save(self, epoch, checkpt, name):
        path = os.path.join(checkpt, name+'.pkl')
        torch.save({'epoch': epoch,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict()},
                    path)

def get_all_rank_scores(model, img_loader, device):
    model.eval()

    score_dict = {}

    with torch.no_grad():
            for img, img_id, _ in img_loader:
                img_id, img = img_id.to(device), img.to(device)
                output = model(img)

                for i, o in zip(img_id, output):
                    score_dict[int(i)] = o.item()
    
    return score_dict

def get_all_rank_scores_lfw(model, img_loader, device):
    model.eval()

    score_dict = {}
    with torch.no_grad():
        for img, img_id, _ in img_loader:
            img_id, img = img_id.to(device), img.to(device)
            output = model(img)

            for i, o in zip(img_id, output):
                score_dict[int(i)] = o
    
    return score_dict

def get_loss1_loss2(model, img_loader, device, data_loader_wo_img):
    model.eval()
    
    loss1 = []
    loss2 = []

    score_dict = get_all_rank_scores(model, img_loader, device)

    with torch.no_grad():
        for img_id1, img_id2, label, _, _, _ in data_loader_wo_img:
            img_id1, img_id2 = img_id1.to(device), img_id2.to(device)
            label = label.to(device)

            for i1, i2, l in zip(img_id1, img_id2, label):
                score_diff = score_dict[int(i1)] - score_dict[int(i2)]
                loss1.append((score_diff - l) ** 2 * 0.5)
                loss2.append((score_diff + l) ** 2 * 0.5)

    return np.array(loss1, dtype=float), np.array(loss2, dtype=float)

def get_loss1_loss2_lfw(model, img_loader, device, data_loader_wo_img):
    model.eval()
    
    loss1 = []
    loss2 = []

    score_dict = get_all_rank_scores_lfw(model, img_loader, device)

    with torch.no_grad():
        for img_id1, img_id2, label, attr_id, _, _ in data_loader_wo_img:
            img_id1, img_id2 = img_id1.to(device), img_id2.to(device)
            label, attr_id = label.to(device), attr_id.to(device)

            for i1, i2, l, a in zip(img_id1, img_id2, label, attr_id):
                score_diff = score_dict[int(i1)][int(a)].item() - score_dict[int(i2)][int(a)].item()
                loss1.append((score_diff - l) ** 2 * 0.5)
                loss2.append((score_diff + l) ** 2 * 0.5)
    
    return np.array(loss1, dtype=float), np.array(loss2, dtype=float)


def get_loss1_loss2_food(model, img_loader, device, data_loader_wo_img):
    model.eval()
    
    loss1 = []
    loss2 = []

    score_dict = get_all_rank_scores_lfw(model, img_loader, device)

    with torch.no_grad():
        for anc, pos, neg, _ in data_loader_wo_img:
            anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)

            for a, p, n in zip(anc, pos, neg):
                output1 = score_dict[int(a)].view(1,-1)
                output2 = score_dict[int(p)].view(1,-1)
                output3 = score_dict[int(n)].view(1,-1)
                l1 = triplet_margin_loss(output1, output2, output3, margin=1.0, p=2)
                l2 = triplet_margin_loss(output1, output3, output2, margin=1.0, p=2)

                loss1.append(l1.item())
                loss2.append(l2.item())
    
    return np.array(loss1, dtype=float), np.array(loss2, dtype=float)