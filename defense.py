import os  
import numpy as np 
# import  minpy.numpy as np 
import tqdm
import argparse
from torchvision import transforms
from torch.utils import model_zoo
import torch
import torch.nn as nn 
from Log import MyLog
from utils.ageDataLoader import load_age
from utils.lfwDataLoader import load_lfw
from utils.foodDataLoader import load_food
from torchvision.models import resnet50
import random
import gc 
from collections import defaultdict

model_urls = {
    'resnet50': os.path.join('models', 'resnet50-19c8e357.pth')
}
class FeatureExtra(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureExtra, self).__init__()

        # output (1000)

        self.model = resnet50(pretrained=False)
        if kwargs['pretrained']:
            print('=====> load pretrained model resnet50')
            self.model.load_state_dict(torch.load(model_urls['resnet50']))

    def forward(self, x):
        return self.model(x)

class Sever:

    def __init__(self, 
                opt_method,
                logger,
                batch_size=32,
                eta=0.01,
                p_def = 0.03,
                num_epochs=3,
                num_rounds=5,
                epsilon=0.05,
                dataform = 'pairwise',
                seed=12345):
        super(Sever, self).__init__()

        assert opt_method in ['SGD', 'Adagrad'], 'check the opt method'

        self.opt_method = opt_method
        self.num_rounds = num_rounds
        self.epsilon = epsilon
        self.p_def = p_def

        assert dataform in ['pairwise', 'triplet']
        self.dataform = dataform

        self.batch_size = batch_size
        self.eta = eta
        self.num_epochs = num_epochs
        self.theta = None

        self.logger = logger

        np.random.seed(seed)

    def attack_and_defense(self, Xdata, Ydata):
        '''
        Xdata and Ydata are numpy array.
        and do not split the train/val/test data
        '''

        is_binary = np.unique(Ydata).shape[0]
        
        k = 1
        
        assert is_binary == 2, 'only support binary classfication!'

        N_train, d = Xdata.shape
        
        active_indices = np.arange(N_train)

        self.theta = np.random.uniform(size=(d, k))

        self.logger.info("======> Defensing!")
        for epoch in range(self.num_rounds):
            
            self.logger.info("=====> round %d" % epoch)
            xs = Xdata[active_indices]
            ys = Ydata[active_indices]
            
            '''
            hinge loss with linear classifier (no bias)
            '''
            _, gradients, _, losses, _ = self.train(xs, ys)

            self.logger.info("=====> current training losses %.4f" % (np.sum(losses) / len(active_indices)))
            
            self.logger.info("=====> filterSimple")
            indices, outlier_scores = self.filterByClass(losses, ys, gradients)
            self.logger.info("=====> filtering %d samples" % (len(ys) - len(indices)))

            if len(indices) == len(active_indices) or len(indices) == 0:
                break
            
            active_indices = active_indices[indices]
        
        # print(Xdata)
        
        # print("=====> calc training loss")
        # _, train_loss, _ = self.nabla_Loss(Xdata, Ydata, self.theta, agg=1)
        # _, train_loss, _ = self.nabla_Loss(Xdata[active_indices], Ydata[active_indices], self.theta, agg=1)

        # self.logger.info("=====> total training loss %.4f" % train_loss)
        mask = np.asarray([0 for i in range(N_train)])
        mask[active_indices] = 1
        return mask
    
    def filterByClass(self, losses, y, gradients):
        n = gradients.shape[0]
        
        allIndices = np.arange(n)
        labels = np.unique(y)

        def filterSimple(g, p, m):
            k = 1
            N_filt = g.shape[0]
            gcentered = (g - m) / np.sqrt(N_filt)

            # _, _, V_p = np.linalg.svd(gcentered)
            _, __, V_p = np.linalg.svd(gcentered, full_matrices=0)
            
            # print(_.shape)
            # projection = V_p[:, :k] 
            projection = V_p[:k, :].T

            # %Scores are the magnitude of the projection onto the top principal component
            scores = np.matmul(gcentered, projection)
            scores = np.sqrt(np.sum(np.square(scores), axis=1))

            # debug
            print(scores.shape)
            print(np.quantile(scores, 1 - p))

            indices = np.arange(N_filt)

            if np.quantile(scores, 1 - p) > 0:
                scores = scores / np.quantile(scores, 1 - p)
                indices = indices[scores <= 1.0]
            else:
                scores = scores / np.max(scores)

            # print(indices.shape)
            return indices, scores

        assert -1 in labels and 1 in labels
        
        n_minus = sum(y == -1)
        n_plus = sum(y == 1)
        re_scores = np.zeros(n)

        self.p_def =  (n_plus + n_minus) * self.epsilon / (self.num_rounds * min(n_minus, n_plus))
        
        # print("self.p_def: ", self.p_def)
        re_indices = []
        for i in labels:

            curIndices = allIndices[y == i]

            # print(curIndices.shape)
            curGradients = gradients[curIndices, :] 
            curMean = np.mean(gradients[curIndices, :], axis=0, keepdims=True) # (1, d)

            curFilteredIndices, curScores = filterSimple(curGradients, 
                                                         self.p_def,
                                                         curMean)
            reindex = allIndices[curIndices]
            if len(curFilteredIndices):
                re_indices.extend(reindex[curFilteredIndices].tolist())

            re_scores[curIndices] = curScores
        
        re_indices.sort()

        return re_indices, re_scores
    
    def train(self, X_train, Y_train):

        is_binary = np.unique(Y_train).shape[0]

        k = 1
        assert is_binary == 2, 'only support binary classfication!'

        N_train, d = X_train.shape

        # theta = np.zeros((d, k))
        theta2 = 1e-4 * np.ones((d, k))

        ids = [i for i in range(N_train)]

        for epoch in range(self.num_epochs):
            np.random.shuffle(ids)

            for t in range(0, N_train, self.batch_size):
                t2 = min(t + self.batch_size, N_train)

                Xb = X_train[ids[t:t2], ]
                Yb = Y_train[ids[t:t2] ]

                g, losses, _ = self.nabla_Loss(Xb, Yb, self.theta, 1)

                # for adagrad
                theta2 = theta2 + np.square(g) # (element-wise)
                if self.opt_method == 'Adagrad':
                    self.theta = self.theta - self.eta * (g / np.sqrt(theta2))
                
                elif self.opt_method == 'SGD':
                    self.theta -= self.eta * g
                else:
                    assert (False), 'no implementation!'
            
            self.logger.info("=====> epoch nabla_Loss!")
            _, _, err_train = self.nabla_Loss(X_train, Y_train, self.theta)

            self.logger.info("Error (epoch %d): %.4f" % (epoch, err_train))


        self.logger.info("=====> total nabla_Loss!")
        gradients, losses, errs = self.nabla_Loss(X_train, Y_train, self.theta, agg=0)
        return self.theta, gradients, theta2, losses, errs

    def nabla_Loss(self, X, y, theta, agg=1):
        
        N, d = X.shape

        # print("X.shape:", X.shape)
        # y = np.reshape(y, (-1, 1))
        yx = np.matmul(np.diag(y.flatten()), X) 
        
        # print(y.shape)
        # print(X.shape)
        # print(theta.shape)

        margins = np.matmul(yx, theta)
        # print(margins.shape)

        losses = np.maximum(1 - margins, 0)
        mults = -1 * (margins < 1) # (N, 1)

        # print(mults.shape)

        errs = (margins < 0) + 0.5 * (margins == 0)

        if agg:
            gradients = (np.matmul(mults.T, yx)).T / N  # (d, 1)

            loss = np.sum(losses) / N

            err = np.sum(errs) / N
        else:
            gradients = np.matmul(np.diag(mults.flatten()), yx)# (N, d)
            loss = losses 
            err = errs

        return gradients, loss, err


def process_food_data(base, clean_data_name):
    
    dicts = defaultdict(dict)

    # print(clean_data_name)

    if not os.path.exists(clean_data_name):
        assert False
    with open(clean_data_name, 'r') as f:
        for line in f.readlines():
            anc, f1, f2, label = line.strip().split(',')
            dicts.setdefault(str(anc) + ',' + str(f1) + ',' + str(f2), int(label))
            dicts.setdefault(str(anc) + ',' + str(f2) + ',' + str(f1), int(label))
    
    dirs = os.listdir(base)
    for dir in dirs:
        # if not dir.startswith('pga'):
        #     continue
        if 'fliped' in dir or not dir.startswith('pga'):
            continue
        write = []
        with open(os.path.join(base, dir), 'r') as f:
            for sam in f.readlines():
                a, p, n, _ = sam.strip().split(',')
                key = str(a) + ',' + str(p) + ',' + str(n)
                if not key in dicts:
                    assert False
                
                write.append(key + ',' + str(dicts[key]))
        
        with open(os.path.join(base, dir[:-4] + '_fliped.txt'), 'w') as f:
            f.writelines('\n'.join(write))


if __name__ == '__main__':
    

    # debug the defense model
    # np.random.seed(12345)
    # N, d = 10000, 10

    # X = np.random.normal(size = (N, d))
    # Y = np.random.uniform(size = (N))

    # Y[Y < 0.5] = -1
    # Y[Y >= 0.5] = 1

    # Y[-1] = 1
    # Y[-2] = -1
    
    # # print(X)
    # # print(y)

    # se = Sever("SGD")

    # se.attack_and_defense(X, y)


    
    '''
    define argument parser
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='food', choices=['age', 'lfw', 'food'])
    parser.add_argument('--flip_type', type=str, default='threshold',
                        choices=['clean', 'random', 'reverse', 'nearest', 'furthest', 'mask', 'threshold', 'pga', 'l2_mask'])
    parser.add_argument('--flip_ratio', type=int, default=25, choices=[0, 15, 25, 35])

    parser.add_argument('--batch_size', type=int, default=32)

    # deep model params
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dim', type=int, default=1)
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_rounds', type=int, default=2)
    parser.add_argument('--num_epochs', type=int, default=3)

    parser.add_argument('--seed', type=int, default=12345)

    args = parser.parse_args()

    # name
    name = '{}_{}'.format(args.flip_type, args.flip_ratio)
    # path
    train_path = 'data/{}/train/{}.txt'.format(args.dataset, name)
    test_path = 'data/{}/test.txt'.format(args.dataset)
    img_path = 'data/{}/imgs/imgs'.format(args.dataset)
    checkpt_dir = 'checkpt/{}'.format(args.dataset)
    log_dir = 'log/{}'.format(args.dataset)

    defense_name = '{}_{}_{}'.format(args.flip_type, args.flip_ratio, 'sever')
    defense_data_dir = 'data/{}/train/{}_new.txt'.format(args.dataset, defense_name)

    '''
    prepair file architecture
    '''
    print('=====> checking file architecture ...')

    if args.flip_type == 'clean' or args.flip_ratio == 0:
        if not args.flip_type.startswith('clean'):
            args.flip_type = 'clean'
        args.flip_ratio = 0

    '''
    logger
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = MyLog(os.path.join(log_dir, name + '.log'))
    
    logger.info(args)

    # check file existence
    if not os.path.exists(train_path):
        logger.info('Error: train file path({}) doesn\'t exist!'.format(train_path))
        exit()
    if not os.path.exists(test_path):
        logger.info('Error: test file path({}) doesn\'t exist!'.format(test_path))
        exit()
    if not os.path.exists(img_path):
        logger.info('Error: image directory({}) doesn\'t exist!'.format(img_path))
        exit()
    
    '''
    fix random seed
    '''
    seed = args.seed
    device = 'cuda:0'

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module

    # print(model)
    # logger.info('=====> processed data...')
        
    # process_food_data(os.path.join('data', args.dataset, 'train'), os.path.join('data', args.dataset, 'train', 'clean_0_fliped.txt'))

    # save the original feature
    if not os.path.exists(os.path.join('data', args.dataset, 'origin')):
        os.makedirs(os.path.join('data', args.dataset, 'origin'))
    
    if not os.path.exists(os.path.join('data', args.dataset, 'origin', name + '.npz')):
        # only need onece
       
        logger.info('=====> loading data ...')

        dicts = {'pretrained': args.pretrained}
        model = FeatureExtra(**dicts).to(args.device)

        others = {'num_workers': 1, 'pin_memory': False}

        transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])]
        )

        if args.dataset == 'food':
            train_path = 'data/{}/train/{}_fliped.txt'.format(args.dataset, name)
        data_params = {
        'train_path': train_path,
        'test_path': test_path,
        'img_path': img_path,
        'transform':transform,
        'shuffle': False,
        'batch_size': args.batch_size,
        'others': others,

        'int2food_path': 'data/food/int2food.pkl'
        }

        data_load_functions = {
        'age': load_age,
        'lfw': load_lfw,
        'food': load_food,
        }

        xx = []
        yy = []
        # extract the feature
        _, train_loader, _, img_loader, _, test_loader = data_load_functions[args.dataset](**data_params)
        with torch.no_grad():
            model.eval()
            if args.dataset in ['age', 'lfw']:
                for data in train_loader:
                    img1, img2, label = data[2].to(device), data[3].to(device), data[4].to(device)
                    x1 = model(img1).cpu().numpy()
                    x2 = model(img2).cpu().numpy()

                    # print(label.reshape(-1, 1).size())
                    # print((x1 - x2).size())
                    xx.append(x1 - x2)
                    yy.append(label.cpu().numpy())
            elif args.dataset == 'food':
                for data in train_loader:
                    anchor, pos, neg, label = data[3].to(device), data[4].to(device), data[5].to(device), data[6].to(device)
                    anchor_fea = model(anchor).cpu().numpy()
                    pos_fea = model(pos).cpu().numpy()
                    neg_fea = model(neg).cpu().numpy()
                    label = label.cpu().numpy()

                    xx.append((np.abs(pos_fea - anchor_fea) - np.abs(neg_fea - anchor_fea)) * np.reshape(label, (-1, 1)))
                    yy.append(label)
                # for data in train_loader:
            else:
                assert False

            X = np.concatenate(xx, axis = 0)
            Y = np.concatenate(yy, axis = 0)

        # print(X.shape)
        # print(Y.shape)

        np.savez(os.path.join('data', args.dataset, 'origin', name + '.npz'), X=X, Y=Y)
    else:
        torch.cuda.empty_cache()
        gc.collect()
        
        logger.info("======> load data from %s" % os.path.join('data', args.dataset, 'origin', name + '.npz'))
        _data = np.load(os.path.join('data', args.dataset, 'origin', name + '.npz'))
        X = _data['X']
        Y = _data['Y']

    # print(X[:5])
    # print(Y[:5])

    # print(X.shape)
    # print(Y.shape)

    logger.info("=====> total samples: %d" % Y.shape[0])
    se = Sever(args.optim,
               logger, 
               batch_size=args.batch_size,
               num_rounds=args.num_rounds,
               num_epochs=args.num_epochs)
    mask = se.attack_and_defense(X, Y)

    assert len(mask) == len(Y), "check the filtering length!"

    Y = Y * mask

    print(Y.shape)
    
    print(Y[:5])

    n_plus = sum(Y == 1)
    n_minus = sum(Y == -1)

    logger.info('=====> after defense, info:')
    logger.info('=====> positive pair: %d' % n_plus)
    logger.info('=====> negative pair: %d' % n_minus)
    logger.info('=====> total pair: %d' % (n_plus + n_minus))



    # logger.info('=====> save data to defense_dir: %s' % defense_data_dir)
    lines = [ _ for _ in open(train_path, 'r')]
    filter_lines = []
    # print(lines[:5])
    assert len(lines) == len(mask), 'lines and mask'
    for line in range(len(lines)):
        if mask[line] != 0:
            filter_lines.append(lines[line])

    assert len(filter_lines) == (n_plus + n_minus)

    with open(defense_data_dir, 'w') as f:
        f.writelines(''.join(filter_lines))



