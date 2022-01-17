import os
import csv
import torch
import random
import timeit
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from audtorch.metrics.functional import concordance_cc
from torch.utils.data import TensorDataset, DataLoader

# process stimuli features
with open('/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_stimuli/sti.csv') as sti:
    file_content = csv.reader(sti, delimiter=',')
    headers = next(file_content, None)
    feats_sti = list(file_content)
feats_sti = np.array(feats_sti, dtype=float)
feats_sti -= np.mean(feats_sti, axis = 0)
feats_sti /= (np.std(feats_sti, axis = 0) + 0.0001)
feats_sti = torch.from_numpy(feats_sti)

# process participant features
feats_train = []
ind_train = []
comp_train = []
warm_train = []
feats_valid = []
ind_valid = []
comp_valid = []
warm_valid = []

path_feats = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/feats_participant/'
path_labels = '/Users/liyuanchao/Documents/Corpus/IMPRESSION/feats_labels/labels/'


os.chdir(path_feats)
for file in range(32): #change number for cross-validation
    with open(str(file) + '.csv') as par:
        file_content = csv.reader(par, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            feats_train.append(row[:-1])
            ind_train.append(row[-1])
for file in range(32, 40): #change number for cross-validation
    with open(str(file) + '.csv') as par:
        file_content = csv.reader(par, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            feats_valid.append(row[:-1])
            ind_valid.append(row[-1])            

os.chdir(path_labels)
for file in range(32): #change number for cross-validation
    with open(str(file) + '.csv') as label:
        file_content = csv.reader(label, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            comp_train.append(row[0])
            warm_train.append(row[1])
for file in range(32, 40): #change number for cross-validation
    with open(str(file) + '.csv') as label:
        file_content = csv.reader(label, delimiter=',')
        headers = next(file_content, None)
        for row in list(file_content):
            comp_valid.append(row[0])
            warm_valid.append(row[1])
            
feats_train = np.array(feats_train, dtype=float)
comp_train = np.array(comp_train, dtype=float)
warm_train = np.array(warm_train, dtype=float)
ind_train = np.array(ind_train, dtype=int)
feats_valid = np.array(feats_valid, dtype=float)
comp_valid = np.array(comp_valid, dtype=float)
warm_valid = np.array(warm_valid, dtype=float)
ind_valid = np.array(ind_valid, dtype=int)

torch.manual_seed(1)

# shuffle data
leng = len(feats_train)
indices = np.arange(leng)
random.shuffle(indices)
feats_train[np.arange(leng)] = feats_train[indices]
comp_train[np.arange(leng)] = comp_train[indices]
warm_train[np.arange(leng)] = warm_train[indices]
ind_train[np.arange(leng)] = ind_train[indices]

# normalization
feats_train -= np.mean(feats_train, axis = 0)
feats_train /= (np.std(feats_train, axis = 0) + 0.0001)
feats_valid -= np.mean(feats_valid, axis = 0)
feats_valid /= (np.std(feats_valid, axis = 0) + 0.0001)

feats_train = torch.from_numpy(feats_train)
feats_valid = torch.from_numpy(feats_valid)
ind_train = torch.from_numpy(ind_train)
ind_valid = torch.from_numpy(ind_valid)
comp_train = torch.from_numpy(comp_train)
comp_valid = torch.from_numpy(comp_valid)
warm_train = torch.from_numpy(warm_train)
warm_valid = torch.from_numpy(warm_valid)

print(feats_train.size(), feats_valid.size(), feats_sti.size())
print(ind_train.size(), ind_valid.size())
print(comp_train.size(), comp_valid.size(), warm_train.size(), warm_valid.size())

trainset = TensorDataset(feats_train, ind_train, comp_train, warm_train)
validset = TensorDataset(feats_valid, ind_valid, comp_valid, warm_valid)
traindata = DataLoader(dataset=trainset, batch_size=64, shuffle=False)
validdata = DataLoader(dataset=validset, batch_size=64, shuffle=False)

print('Data preparation completed!')

# model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.lstm1 = nn.LSTM(input_size=len(feats_train[0]),
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=len(feats_sti[0]),
                            hidden_size=64,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.attn = nn.MultiheadAttention(128, 16, batch_first=True)
        self.drop = nn.Dropout(p=0.5)
        self.dense = nn.Linear(128, 16)
        self.acti = nn.ReLU()
        self.out = nn.Linear(16, 1)

    def forward(self, input_par, input_sti):
        # lstm
        self.lstm1.flatten_parameters()
        self.lstm2.flatten_parameters()
        x_par, _ = self.lstm1(input_par)
        x_sti, _ = self.lstm2(input_sti)
        # attention
        x_par, _ = self.attn(x_par, x_par, x_par)
        x_sti, _ = self.attn(x_sti, x_sti, x_sti)
        x_par_sti, _ = self.attn(x_par, x_sti, x_sti)
        x_sti_par, _ = self.attn(x_sti, x_par, x_par)

        # distillation loss
        loss_dis1 = func(x_par, x_sti)
        loss_dis2 = func(x_sti, x_par)
        # similarity loss
        loss_sim1 = kl_func(nn.functional.log_softmax(x_par_sti, 0), nn.functional.softmax(x_par, 0))
        loss_sim2 = kl_func(nn.functional.log_softmax(x_sti_par, 0), nn.functional.softmax(x_sti, 0))

        # concatenation
        x_co = torch.cat((x_par, x_sti, x_par_sti, x_sti_par), 1)
        x_co = self.drop(x_co)
        x_co = x_co.mean(dim=1)  # pooling
        x_co = self.dense(x_co)
        x_co = self.acti(x_co)
        comp = self.out(x_co)
        warm = self.out(x_co)
        return comp, warm, loss_dis1, loss_dis2, loss_sim1, loss_sim2
      
model = NeuralNet()
model = model.to(torch.float64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
func = nn.MSELoss()
kl_func = nn.KLDivLoss(reduction='batchmean')

cross_loss_train = []
sim_loss_train = []
dis_loss_train = []
cross_loss_valid = []
sim_loss_valid = []
dis_loss_valid = []

# training
for epoch in range(50):
    start = timeit.default_timer()
    print("-----epoch: ", epoch, "-----")
    comp_loss_list_train = []
    comp_loss_list_valid = []
    warm_loss_list_train = []
    warm_loss_list_valid = []
    cross_loss_list_train = []
    cross_loss_list_valid = []
    sim_loss_list_train = []
    sim_loss_list_valid = []
    dis_loss_list_train = []
    dis_loss_list_valid = []
    comp_preds_train = []
    comp_preds_valid = []
    warm_preds_train = []
    warm_preds_valid = []
    print('--training begins--')
    model.train()
    for input_par, inds, labels_comp, labels_warm in traindata:
        input_sti = torch.tensor([])
        for inde in inds:
            input_sti = torch.cat((input_sti, feats_sti[inde]), 0)
        input_par = input_par.reshape(input_par.shape[0], 1, input_par.shape[1])
        input_sti = input_sti.reshape(input_par.shape[0], 1, -1)    
        # loss
        preds_comp, preds_warm, loss_dis1, loss_dis2, loss_sim1, loss_sim2 = model(input_par, input_sti)
        train_loss_comp = func(preds_comp.squeeze(), labels_comp)
        train_loss_warm = func(preds_warm.squeeze(), labels_warm)
        loss_sim = loss_sim1 + loss_sim2
        loss_dis = loss_dis1 + loss_dis2
        train_cross_loss = loss_dis + loss_sim
        comp_loss_list_train.append(train_loss_comp.item())
        warm_loss_list_train.append(train_loss_warm.item())
        cross_loss_list_train.append(train_cross_loss.item())
        sim_loss_list_train.append(loss_sim.item())
        dis_loss_list_train.append(loss_dis.item())
        for i in preds_comp:
            comp_preds_train.append(i.item())
        for i in preds_warm:
            warm_preds_train.append(i.item())
        # backprop
        optimizer.zero_grad()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        train_loss = train_loss_comp + train_loss_warm + train_cross_loss
        train_loss.backward()
        optimizer.step()
    print('--training ends--')

# validation
    print('--validation begins--')
    model.eval()
    input_par = feats_valid
    for input_par, inds, labels_comp, labels_warm in validdata:
        input_sti = torch.tensor([])
        for inde in inds:
            input_sti = torch.cat((input_sti, feats_sti[inde]), 0)
        input_par = input_par.reshape(input_par.shape[0], 1, input_par.shape[1])
        input_sti = input_sti.reshape(input_par.shape[0], 1, -1)
        # loss
        preds_comp, preds_warm, loss_dis1, loss_dis2, loss_sim1, loss_sim2 = model(input_par, input_sti)
        valid_loss_comp = func(preds_comp.squeeze(), labels_comp)
        valid_loss_warm = func(preds_warm.squeeze(), labels_warm)
        loss_sim = loss_sim1 + loss_sim2
        loss_dis = loss_dis1 + loss_dis2
        valid_cross_loss = loss_dis + loss_sim
        comp_loss_list_valid.append(valid_loss_comp.item())
        warm_loss_list_valid.append(valid_loss_warm.item())
        cross_loss_list_valid.append(valid_cross_loss.item())
        sim_loss_list_valid.append(loss_sim.item())
        dis_loss_list_valid.append(loss_dis.item())
        for i in preds_comp:
            comp_preds_valid.append(i.item())
        for i in preds_warm:
            warm_preds_valid.append(i.item())
            
    # compute performance for each epoch
    comp_preds_train = torch.tensor(comp_preds_train)
    warm_preds_train = torch.tensor(warm_preds_train)
    comp_preds_valid = torch.tensor(comp_preds_valid)
    warm_preds_valid = torch.tensor(warm_preds_valid)

    train_ccc_comp = concordance_cc(comp_preds_train, comp_train)
    train_ccc_warm = concordance_cc(warm_preds_train, warm_train)
    valid_ccc_comp = concordance_cc(comp_preds_valid, comp_valid)
    valid_ccc_warm = concordance_cc(warm_preds_valid, warm_valid)

    train_loss_comp = sum(comp_loss_list_train) / len(comp_loss_list_train)
    train_loss_warm = sum(warm_loss_list_train) / len(warm_loss_list_train)
    valid_loss_comp = sum(comp_loss_list_valid) / len(comp_loss_list_valid)
    valid_loss_warm = sum(warm_loss_list_valid) / len(warm_loss_list_valid)
    train_loss_cross = sum(cross_loss_list_train) / len(cross_loss_list_train)
    valid_loss_cross = sum(cross_loss_list_valid) / len(cross_loss_list_valid)
    train_loss_sim = sum(sim_loss_list_train) / len(sim_loss_list_train)
    train_loss_dis = sum(dis_loss_list_train) / len(dis_loss_list_train)
    valid_loss_sim = sum(sim_loss_list_valid) / len(sim_loss_list_valid)
    valid_loss_dis = sum(dis_loss_list_valid) / len(dis_loss_list_valid)
    
    cross_loss_train.append(train_loss_cross)
    sim_loss_train.append(train_loss_sim)
    dis_loss_train.append(train_loss_dis)
    cross_loss_valid.append(valid_loss_cross)
    sim_loss_valid.append(valid_loss_sim)
    dis_loss_valid.append(valid_loss_dis)
        
    print('train_loss_comp: %.4f' % train_loss_comp, '|train_ccc_comp: %.4f' % train_ccc_comp, '\n'
          'train_loss_warm: %.4f' % train_loss_warm, '|train_ccc_warm: %.4f' % train_ccc_warm, '\n'
          'valid_loss_comp: %.4f' % valid_loss_comp, '|valid_ccc_comp: %.4f' % valid_ccc_comp, '\n'
          'valid_loss_warm: %.4f' % valid_loss_warm, '|valid_ccc_warm: %.4f' % valid_ccc_warm, '\n'
          'train_loss_cross: %.4f' % train_loss_cross, '|valid_loss_cross: %.4f' % valid_loss_cross, '\n'
          'train_loss_sim: %.4f' % train_loss_sim, '|valid_loss_sim: %.4f' % valid_loss_sim, '\n'
          'train_loss_dis: %.4f' % train_loss_dis, '|valid_loss_dis: %.4f' % valid_loss_dis)

    print('---validation ends---')

    stop = timeit.default_timer()
    print('Time: ', stop - start)
    scheduler.step()
    
# plot loss fig
plt.plot(sim_loss_train, label="train", color="red")
plt.plot(sim_loss_valid, label="valid", color="blue")
plt.legend()
plt.title('Similarity enhancement loss')
plt.savefig('/Users/liyuanchao/Desktop/sim.png', dpi=400)
plt.show()

plt.plot(dis_loss_train, label="train", color="red")
plt.plot(dis_loss_valid, label="valid", color="blue")
plt.legend()
plt.title('Knowledge distillation loss')
plt.savefig('/Users/liyuanchao/Desktop/dis.png', dpi=400)
plt.show()
