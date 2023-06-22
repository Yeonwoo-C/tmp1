#!/usr/bin/env python
# coding: utf-8

# In[1]:

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score
import math
# import matplotlib.pyplot as plt
import sys
import pandas as pd
from torch import linalg as LA
import torch.utils.data
from torch.utils.data.dataset import Dataset
import copy
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.sampler import SequentialSampler
import random
from math import floor
import torch.nn.functional as F
from torch.nn import init
from sklearn.metrics import roc_auc_score
from functools import reduce
from sklearn.metrics import accuracy_score
import tqdm
np.seterr(divide='ignore', invalid='ignore')
import os
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind
from sklearn import svm
import sys
from functools import reduce
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import argparse
import sklearn.metrics as metrics
from torch.autograd import Variable
import itertools
import xgboost
torch.manual_seed(111)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def feature_selection(X, y):
    data_label1 = np.asarray([X[i] for i in range(len(y)) if y[i] == 1])
    data_label0 = np.asarray([X[i] for i in range(len(y)) if y[i] == 0])
    p = ttest_ind(data_label1, data_label0)[1]
    keep_ttest_index = np.argsort(p)[0:500]  # np.where(p < .001)[0]
    return keep_ttest_index


def load_data(path):
    data = pd.read_csv(path, delimiter=',', index_col=0)
    cols = data.columns.tolist()
    data = np.log1p(data)
    data.loc[:, 'var'] = data.loc[:, cols].var(axis=1)
    drop_index = data[data['var'] < 0].index.tolist()
    data.drop(index=drop_index, inplace=True)
    X = data[cols]

    return X



def prediction(Real, Fake, GAN_epoch,train_index1,valid_index1,labels):
    x = np.array(Real).astype(float)
    #
    train_y = labels[train_index1]
    train_x = x[train_index1,:]
    #print(train_x.shape)
    train_x = np.concatenate((train_x,Fake))
    #print(train_x.shape)
    valid_x = x[valid_index1,:]
    valid_y = labels[valid_index1]
    train_y = np.concatenate((train_y,train_y))
    #print(len(train_y))

    #clf = svm.SVC(kernel='linear',probability=True).fit(X,train_y)
    clf = xgboost.XGBRegressor(tree_method='gpu_hist')
    clf.fit(train_x,train_y)
    y_pred = clf.predict(valid_x)
    fpr, tpr, threshold = metrics.roc_curve(valid_y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)

    return roc_auc



class Discriminator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            #nn.Dropout(0.3),

            nn.Linear(1024, 50),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            # nn.Linear(1024, 768),
            # nn.BatchNorm1d(768),
            # nn.ReLU(),

            # nn.Linear(768, 512),
            # nn.BatchNorm1d(512),
            # nn.ReLU(),

            nn.Linear(512, n_input),
        )

    def forward(self, x):
        output = self.model(x)

        return output


# In[2]:


def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


# In[3]:
def pre_omics(genB,disB,dat1, dat2,train_index1,valid_index1,labels,  fold, lr_g_tmp,lr_d_tmp,c_a_tmp,c_b_tmp,p_tmp):
    #print('Generating mRNA update ' + str(update))
    #index_ = np.concatenate([train_index1,valid_index1])
    index_ = train_index1

    n_input_1 = np.size(dat1, 1)
    sample_size = np.size(dat1, 0)
    n_input_2 = np.size(dat2, 0)

    # C = np.sqrt(np.outer(np.sum(np.absolute(adj), 0), np.sum(np.absolute(adj), 1)))
    # adj = np.divide(adj, C.transpose())
    dat1_train_loader = torch.utils.data.DataLoader(torch.from_numpy(dat1),
                                                    batch_size=sample_size, shuffle=False)


    num_epochs = 1000
    critic_ite = 5
    weight_clip = 0.01


    optimizer_G = torch.optim.Adam(genB.parameters(),lr = lr_g_tmp)
    optimizer_D = torch.optim.Adam(disB.parameters(),lr=lr_d_tmp)

    best = None
    counter = 0




    for epoch in range(num_epochs):
        #if epoch % 200 == 0:
        #    print("Epoch:", epoch)
        for n, real_samples  in enumerate(dat1_train_loader):

            dat2_train_data = dat2[:, n * sample_size:(n + 1) * sample_size]
            real_all_B = torch.from_numpy(dat2.transpose())

            dat2_train_data = dat2[:,index_]#
            dat2_train_data = torch.from_numpy(dat2_train_data).to(device)

            latent_B = real_samples[index_,:].to(device)#



            real_A = real_samples[index_,:].to(device)#
            real_B = dat2_train_data.t().to(device)

            valid = Variable(torch.tensor(np.ones((real_B.size(0), 1))), requires_grad=False).float().to(device)
            fake = Variable(torch.tensor(np.zeros((real_B.size(0), 1))), requires_grad=False).float().to(device)

            # Forward
            fake_B = genB(latent_B)




            # Discriminator
            for param in disB.parameters():
                param.requires_grad = True
            optimizer_D.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = disB(fake_AB.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake.expand_as(pred_fake))

            real_AB = torch.cat((real_A,real_B),1)
            pred_real = disB(real_AB)
            loss_D_real = criterion_GAN(pred_real, valid.expand_as(pred_real))

            loss_D = (loss_D_fake + loss_D_real)*0.5
            loss_D.backward()
            optimizer_D.step()

            # Generator
            for param in disB.parameters():
                param.requires_grad = False
            optimizer_G.zero_grad()
            fake_AB = torch.cat((real_A,fake_B),1)
            pred_fake = disB(fake_AB)
            loss_G_GAN = criterion_GAN(pred_fake,valid.expand_as(pred_fake))
            loss_G_L1 = criterion_L1(fake_B,real_B)
            loss_G = c_a_tmp*loss_G_GAN + c_b_tmp*loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            #print("predicting")
            res = prediction(real_all_B,fake_B.cpu().detach().numpy(), epoch, train_index1,valid_index1,labels)
            #res = loss_G_L1
            #print(res)

            if best is None:
                best = res

            else:
                if best < res:
                    best = res
                    counter = 0
                else:
                    counter = counter + 1
            if counter == p_tmp:
                return best


    return best


def generate_omics(x1, x2, train_index1, valid_index1, labels, fold, filename,filename_):
    index_ = train_index1
    sample_name = x1.columns
    feature_name = x2.index

    dat1 = np.array(x1).transpose().astype(np.float32)
    dat2 = np.array(x2).astype(np.float32)
    n_input_1 = np.size(dat1, 1)
    sample_size = np.size(dat1, 0)
    n_input_2 = np.size(dat2, 0)

    # C = np.sqrt(np.outer(np.sum(np.absolute(adj), 0), np.sum(np.absolute(adj), 1)))
    # adj = np.divide(adj, C.transpose())
    dat1_train_loader = torch.utils.data.DataLoader(torch.from_numpy(dat1),
                                                    batch_size=sample_size, shuffle=False)

    par_set = list(itertools.product(lr_g_list, lr_d_list, c_a_list, c_b_list,p_list))
    count = 0
    for par_search in tqdm.tqdm(range(0,len(par_set))):
        par_tmp = par_set[par_search]
        lr_g_tmp = par_tmp[0]
        lr_d_tmp = par_tmp[1]
        c_a_tmp = par_tmp[2]
        c_b_tmp = par_tmp[3]
        p_tmp = par_tmp[4]

        discriminatorB = Discriminator(2 * n_input_2).to(device)
        generatorB = Generator(n_input_2).to(device)

        discriminatorB.apply(weights_init)
        generatorB.apply(weights_init)

        res_loss = pre_omics(generatorB, discriminatorB, dat1, dat2, train_index1, valid_index1, labels,  fold,lr_g_tmp,lr_d_tmp,c_a_tmp,c_b_tmp,p_tmp)


        if count == 0:
            best_loss = res_loss
            lr_G = lr_g_tmp
            lr_D = lr_d_tmp
            c_a = c_a_tmp
            c_b = c_b_tmp
            patience = p_tmp
            count = count + 1

        else:
            if best_loss < res_loss:
                best_loss = res_loss
                lr_G = lr_g_tmp
                lr_D = lr_d_tmp
                c_a = c_a_tmp
                c_b = c_b_tmp
                patience = p_tmp


    print("Search finish")


    discriminatorB = Discriminator(2 * n_input_2).to(device)
    generatorB = Generator(n_input_2).to(device)

    discriminatorB.apply(weights_init)
    generatorB.apply(weights_init)

    num_epochs = 1000
    critic_ite = 5
    weight_clip = 0.01

    optimizer_G = torch.optim.Adam(generatorB.parameters(), lr=lr_G)
    optimizer_D = torch.optim.Adam(discriminatorB.parameters(), lr=lr_D)

    best = None
    best2 = None
    counter = 0
    counter2 = 0

    for epoch in range(num_epochs):
        #if epoch % 200 == 0:
        #    print("Epoch:", epoch)
        for n, real_samples in enumerate(dat1_train_loader):
            real_all_B = torch.from_numpy(dat2.transpose())
            dat2_train_data = dat2[:, n * sample_size:(n + 1) * sample_size]
            dat2_train_data = torch.from_numpy(dat2_train_data[:,index_]).to(device)

            latent_B = real_samples[index_,:].to(device)

            real_A = real_samples[index_,:].to(device)
            real_B = dat2_train_data.t().to(device)

            valid = Variable(torch.tensor(np.ones((real_B.size(0), 1))), requires_grad=False).float().to(device)
            fake = Variable(torch.tensor(np.zeros((real_B.size(0), 1))), requires_grad=False).float().to(device)

            # Forward
            fake_B = generatorB(latent_B)

            latent_all = real_samples.to(device)
            fake_all = generatorB(latent_all)

            # Discriminator
            for param in discriminatorB.parameters():
                param.requires_grad = True
            optimizer_D.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = discriminatorB(fake_AB.detach())
            loss_D_fake = criterion_GAN(pred_fake, fake.expand_as(pred_fake))

            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = discriminatorB(real_AB)
            loss_D_real = criterion_GAN(pred_real, valid.expand_as(pred_real))

            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # Generator
            for param in discriminatorB.parameters():
                param.requires_grad = False
            optimizer_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = discriminatorB(fake_AB)
            loss_G_GAN = criterion_GAN(pred_fake, valid.expand_as(pred_fake))
            loss_G_L1 = criterion_L1(fake_B, real_B)
            loss_G = c_a*loss_G_GAN + c_b*loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            res = prediction(real_all_B, fake_B.cpu().detach().numpy(), epoch, train_index1, valid_index1, labels)
            #res = loss_G_L1
            # print(auc)

            if counter < patience:
                if best is None:
                    best = res
                    save_ = fake_all#

                else:
                    if best < res:
                        best = res
                        save_ = fake_all#
                        counter = 0
                    else:
                        counter = counter + 1

            ##### Second generated
            if best2 is None:
                best2 = loss_G
                save_2 = fake_all
            else:
                if best2 > loss_G:
                    best2 = loss_G
                    save_2 = fake_all
                    counter2 = 0
                else:
                    counter2 = counter2 + 1
            if counter2==patience and counter==patience:
                save_ = save_.cpu().detach().numpy()
                dd = pd.DataFrame(save_, index=sample_name, columns=feature_name)
                dd.to_csv(filename)
                save_2 = save_2.cpu().detach().numpy()
                dd2 = pd.DataFrame(save_2, index = sample_name, columns=feature_name)
                dd2.to_csv(filename_)
                return best, lr_G, lr_D, c_a, c_b, patience

    save_ = save_.cpu().detach().numpy()
    dd = pd.DataFrame(save_, index=sample_name, columns=feature_name)
    dd.to_csv(filename)
    save_2 = save_2.cpu().detach().numpy()
    dd2 = pd.DataFrame(save_2, index=sample_name, columns=feature_name)
    dd2.to_csv(filename_)
    return best,lr_G,lr_D,c_a,c_b,patience




# In[4]:


#from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
#kf = KFold(n_splits = 5, shuffle = True, random_state = 10)

x1 = "./mRNA.csv"
x2 = "./meth.csv"

label = "./label.csv"

x1 = pd.read_csv(x1, index_col=0, delimiter=',')
x2 = pd.read_csv(x2, index_col=0, delimiter=',')
xy, x_ind, y_ind = np.intersect1d(x1.columns, x2.columns, return_indices=True)
#_, x_ind1, y_ind1 = np.intersect1d(x2.index, adj.columns, return_indices=True)
#xy1, x_ind2, y_ind2 = np.intersect1d(x1.index, adj.index, return_indices=True)

x1 = x1.iloc[:, x_ind]
x2 = x2.iloc[:, y_ind]
#x1 = x1.iloc[x_ind2, :]
#x2 = x2.iloc[x_ind1, :]
#adj = adj.iloc[:, y_ind1]
#adj = adj.iloc[y_ind2, :]
x1 = x1.fillna(0)
x2 = x2.fillna(0)

data = pd.read_csv(label, delimiter=',', index_col=0)
xy, x_ind, y_ind = np.intersect1d(x1.columns, data.index, return_indices=True)
x1 = x1.iloc[:, x_ind]
x2 = x2.iloc[:, x_ind]
y = data.iloc[y_ind, :].astype(str)
# y[y=='Positive']=1
# y[y=='Negative']=0
labels = np.array(y).astype(np.float32)
labels = labels[:,1]

criterion_identity = torch.nn.L1Loss()
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_L1 = torch.nn.L1Loss()




#lr_d_list = [0.05, 0.005]
#lr_g_list = [0.05, 0.005]
lr_d_list = [0.001, 0.0005]
lr_g_list = [0.001, 0.0005]

c_a_list = [0.5, 1.0]
c_b_list = [10.0, 20.0,50.0]
p_list = [10,30]
param1_list = []
param2_list = []
for fold in range(1, 6):
    print("Fold:",fold)
    train_index1 = pd.read_csv("./train_index"+str(fold)+".csv",header=None)
    valid_index1 = pd.read_csv("./valid_index"+str(fold)+".csv",header=None)


    train_index1 = np.array(train_index1).ravel().astype(int)
    valid_index1 = np.array(valid_index1).ravel().astype(int)






    #print(adj.shape)
    filename1 = 'omics1_' + str(fold) +  '.csv'
    filename1_ = 'omics1_' + str(fold) + '_loss.csv'
    filename2 = 'omics2_' + str(fold) +  '.csv'
    filename2_ = 'omics2_' + str(fold) + '_loss.csv'
    # Generate omics1
    res2,par1,par2,par3,par4,par5 = generate_omics(x1, x2, train_index1, valid_index1, labels, fold, filename2,filename2_)
    param2_list.append([par1,par2,par3,par4,par5])
    # Generate omics2
    res1,par6,par7,par8,par9,par10 = generate_omics(x2, x1, train_index1, valid_index1, labels, fold, filename1,filename1_)
    param1_list.append([par6,par7,par8,par9,par10])
    #res1, res2 = omics1(x1, x2, adj, labels, train_index1, valid_index1,test_index1, fold)
    print("Omics1:",res1,"Omics2:", res2)

print("Omics1 (lr_g,lr_d,coef1,coef2,patience)")
for i in range(1,(len(param1_list)+1)):
    print("Fold"+str(i)+": ",param1_list[i-1])
print("Omics2 (lr_g,lr_d,coef1,coef2,patience)")
for i in range(1,(len(param2_list)+1)):
    print("Fold"+str(i)+": ",param2_list[i-1])




