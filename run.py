#!/usr/bin/env python

import os
import json
import pprint as pp

import torch
print(torch.__version__)
import torch.optim as optim
from tensorboard_logger import Logger as TbLogger

from options import get_options
from meta_train import train_epoch, validate, get_inner_model, eval, LOO

from reinforce_baselines import  RolloutBaseline
from nets.attention_model_meta import AttentionModel
import collections
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from nets.vaal_model import VAE, Discriminator, AAE
import torch.utils.data.sampler  as sampler
import torch.utils.data as Data
import random
from solver import Solver
from nets.LSTM_Net import LSTM_Net
from nets.Transformer_model import CNNTransmodel


psdata_filename = {'1':'ways_3\\psdata_1.txt','2':'ways_3\\psdata_2.txt', '3':'ways_3\\psdata_3.txt', '4':'ways_3\\psdata_4.txt', '5':'ways_3\\psdata_5.txt', 
'6':'ways_3\\psdata_6.txt', '7':'ways_3\\psdata_7.txt','8':'\\ways_3\\psdata_8.txt', '9':'ways_3\\psdata_9.txt', '10':'ways_3\\psdata_10.txt', '11':'ways_3\\psdata_11.txt', 
'12':'ways_3\\psdata_12.txt', '13':'ways_3\\psdata_13.txt', '14':'ways_3\\psdata_14.txt', '15':'ways_3\\psdata_15.txt', '16':'ways_3\\psdata_16.txt', '17':'ways_3\\psdata_17.txt', 
'18':'ways_3\\psdata_18.txt', '19':'ways_3\\psdata_19.txt', '20':'ways_3\\psdata_20.txt', '21':'ways_3\\psdata_21.txt', '22':'ways_3\\psdata_22.txt', '23':'ways_3\\psdata_23.txt', 
'24':'ways_3\\psdata_24.txt', '25':'ways_3\\psdata_25.txt', '26':'ways_3\\psdata_26.txt', '27':'ways_3\\psdata_27.txt', '28':'ways_3\\psdata_28.txt', '29':'ways_3\\psdata_29.txt', 
'30':'ways_3\\psdata_30.txt', '31':'ways_3\\psdata_31.txt', '32':'ways_3\\psdata_32.txt', '33':'ways_3\\psdata_33.txt', '34':'ways_3\\psdata_34.txt', '35':'ways_3\\psdata_35.txt', 
'36':'ways_3\\psdata_36.txt', '37':'ways_3\\psdata_37.txt', '38':'ways_3\\psdata_38.txt', '39':'ways_3\\psdata_39.txt', '40':'ways_3\\psdata_40.txt', '41':'ways_3\\psdata_41.txt', 
'42':'ways_3\\psdata_42.txt', '43':'ways_3\\psdata_43.txt', '44':'ways_3\\psdata_44.txt', '45':'ways_3\\psdata_45.txt', '46':'ways_3\\psdata_46.txt', }
label_filename = {'1':'ways_3\\label_1.txt','2':'ways_3\\label_2.txt', '3':'ways_3\\label_3.txt', '4':'ways_3\\label_4.txt', '5':'ways_3\\label_5.txt', 
'6':'ways_3\\label_6.txt', '7':'ways_3\\label_7.txt','8':'ways_3\\label_8.txt', '9':'ways_3\\label_9.txt', '10':'ways_3\\label_10.txt', '11':'ways_3\\label_11.txt', 
'12':'ways_3\\label_12.txt', '13':'ways_3\\label_13.txt', '14':'ways_3\\label_14.txt', '15':'ways_3\\label_15.txt', '16':'ways_3\\label_16.txt', '17':'ways_3\\label_17.txt', 
'18':'ways_3\\label_18.txt', '19':'ways_3\\label_19.txt', '20':'ways_3\\label_20.txt', '21':'ways_3\\label_21.txt', '22':'ways_3\\label_22.txt', '23':'ways_3\\label_23.txt', 
'24':'ways_3\\label_24.txt', '25':'ways_3\\label_25.txt', '26':'ways_3\\label_26.txt', '27':'ways_3\\label_27.txt', '28':'ways_3\\label_28.txt', '29':'ways_3\\label_29.txt', 
'30':'ways_3\\label_30.txt', '31':'ways_3\\label_31.txt', '32':'ways_3\\label_32.txt', '33':'ways_3\\label_33.txt', '34':'ways_3\\label_34.txt', '35':'ways_3\\label_35.txt', 
'36':'ways_3\\label_36.txt', '37':'ways_3\\label_37.txt', '38':'ways_3\\label_38.txt', '39':'ways_3\\label_39.txt', '40':'ways_3\\label_40.txt', '41':'ways_3\\label_41.txt', 
'42':'ways_3\\label_42.txt', '43':'ways_3\\label_43.txt', '44':'ways_3\\label_44.txt', '45':'ways_3\\label_45.txt', '46':'ways_3\\label_46.txt', }
sdb_file = {'1':'data\\hba\\02_Sdb.csv', '2':'data\\hba\\03_Sdb.csv', '3':'data\\hba\\04_Sdb.csv', '4':'data\\hba\\05_Sdb.csv', '5':'data\\hba\\07_Sdb.csv', '6':'data\\hba\\08_Sdb.csv',
'7':'data\\hba\\09_Sdb.csv', '8':'data\\hba\\10_Sdb.csv', '9':'data\\hba\\13_Sdb.csv', '10':'data\\hba\\15_Sdb.csv'}
l_file = {'1':'data\\hba\\02_l.csv', '2':'data\\hba\\03_l.csv', '3':'data\\hba\\04_l.csv', '4':'data\\hba\\05_l.csv', '5':'data\\hba\\07_l.csv', '6':'data\\hba\\08_l.csv',
'7':'data\\hba\\09_l.csv', '8':'data\\hba\\10_l.csv', '9':'data\\hba\\13_l.csv', '10':'data\\hba\\15_l.csv'}




def plot_confusion_matrix(cm, labels_name, title):
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))    
    plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
    for first_index in range(len(cm)):  # 第几行
        for second_index in range(len(cm[first_index])):  # 第几列
            temp = cm[first_index][second_index]
            plt.text(second_index, first_index, int(temp), va='center',
                    ha='center',
                    fontsize=13.5)

    plt.xlabel('True label')    
    plt.ylabel('Predicted label')


def run(opts):

    # Pretty print the run args
    pp.pprint(vars(opts))

    # Set the random seed
    torch.manual_seed(opts.seed)

    # Optionally configure tensorboard
    tb_logger = None
    if not opts.no_tensorboard:
        tb_logger = TbLogger(os.path.join(opts.log_dir, "{}".format(opts.problem), opts.run_name))

    os.makedirs(opts.save_dir)
    # Save arguments so exact configuration can always be found
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # Set the device
    opts.device = torch.device("cuda:0" if opts.use_cuda else "cpu")
    opts.cuda = True
    # opts.device = 'cpu'
    

    # Initialize model
    model = AttentionModel(
        opts.input_dim,
        opts.embedding_dim,
        opts.hidden_dim,
        opts.output_dim,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
    ).to(opts.device)

    model_2 = AttentionModel(
        opts.input_dim,
        opts.embedding_dim,
        opts.hidden_dim,
        opts.output_dim,
        n_encode_layers=opts.n_encode_layers,
        mask_inner=True,
        mask_logits=True,
        normalization=opts.normalization,
        tanh_clipping=opts.tanh_clipping,
    ).to(opts.device)


    if opts.use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    




    
   
    '''
    _test_data = pd.read_csv(sdb_file[str(10)],header = None)
    _test_label = pd.read_csv(l_file[str(10)],header = None)
    _test_data = np.array(_test_data).T
    # test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data))
    _test_label = np.array(_test_label)
    _test_label = _test_label.squeeze()
    test_data = []
    test_label = []
    for i in range(len(_test_label)-10):
        test_data.append(_test_data[i:i+10])
        test_label.append(_test_label[i+9])

    _val_data = pd.read_csv(sdb_file[str(6)],header = None)
    _val_label = pd.read_csv(l_file[str(6)],header = None)
    _val_data = np.array(_val_data).T
    # val_data = (val_data-np.min(val_data))/(np.max(val_data)-np.min(val_data))
    _val_label = np.array(_val_label)
    _val_label = _val_label.squeeze()
    val_data = []
    val_label = []
    for i in range(len(_val_label)-10):
        val_data.append(_val_data[i:i+10])
        val_label.append(_val_label[i+9])
    
    train_data = []
    train_label = []
    subject_index = []
    for i in [ 1, 2, 3, 4, 5, 7, 8, 9]:
        data = pd.read_csv(sdb_file[str(i)], header=None)
        label = pd.read_csv(l_file[str(i)], header=None)
        data = np.array(data).T
        label = np.array(label).squeeze()
        for j in range(len(label)-10):
            train_data.append(data[j:j+10])
            train_label.append(label[j+9])
            subject_index.append(i)
    
    
    # train_data = np.load('data/psd/psd_1_trainx.npy')[:,:,7,:]
    # train_label = np.load('data/psd/psd_1_trainy.npy')
    # test_data = np.load('data/psd/psd_1_testx.npy')[:,:,7,:]
    # test_label = np.load('data/psd/psd_1_testy.npy')
    train_data = torch.from_numpy(np.array(train_data)).to(torch.float32)
    train_label = torch.from_numpy(np.array(train_label)).long()
    test_data = torch.from_numpy(np.array(test_data)).to(torch.float32)
    test_label = torch.from_numpy(np.array(test_label)).long()
    subject_index = torch.from_numpy(np.array(subject_index)).long()
    val_data = torch.from_numpy(np.array(val_data)).to(torch.float32)
    val_label = torch.from_numpy(np.array(val_label)).long()

    train_index = np.asarray([i for i in range(train_data.shape[0])])
    np.random.shuffle(train_index)
    # val_index, train_index = train_test_split(index,train_size=0.1,random_state=1234)
    train_index = torch.tensor(train_index)
    # val_index = torch.tensor(val_index)
    # val_data = torch.index_select(train_data, 0, val_index)
    # val_label = torch.index_select(train_label, 0, val_index)
    train_data = torch.index_select(train_data, 0, train_index)
    train_label = torch.index_select(train_label, 0, train_index)
    subject_index = torch.index_select(subject_index, 0, train_index)

    print(train_data.shape, val_data.shape, test_data.shape)
    '''

    
    path = 'data\\FP1\\'
    train_data = []
    train_label = []
    train_label_2 = []
    subject_index = []
    for i in [11, 12, 13, 14, 15, 17, 19, 20, 25, 28, 29]:
        data = np.array(pd.read_csv(path+psdata_filename[str(i)], header=None))
        label = np.array(pd.read_csv(path+label_filename[str(i)], header=None)).reshape(-1)
        for j in range(data.shape[0]-20):
            train_data.append(data[j:j+20])
            train_label.append(label[j+19])
            subject_index.append(i)
    
    for i in range(len(train_label)):                             
        if train_label[i]<2:
            train_label_2.append(0)
        else:
            train_label_2.append(1)
         
    
    train_data = torch.from_numpy(np.array(train_data)).to(torch.float32)
    train_label = torch.from_numpy(np.array(train_label)).long()
    train_label_2 = torch.from_numpy(np.array(train_label_2)).long()
    subject_index = torch.from_numpy(np.array(subject_index)).long()
    

    _test_data = np.array(pd.read_csv(path+psdata_filename[str(31)], header=None))
    _test_label = np.array(pd.read_csv(path+label_filename[str(31)], header=None)).reshape(-1,)
    test_data = []
    test_label = []
    test_label_2 = []
    for i in range(_test_data.shape[0]-20):
        test_data.append(_test_data[i:i+20])
        test_label.append(_test_label[i+19])

    for i in range(len(test_label)):
        if test_label[i] < 2:
            test_label_2.append(0)
        else:
            test_label_2.append(1)

    test_data = torch.from_numpy(np.array(test_data)).to(torch.float32)
    test_label = torch.from_numpy(np.array(test_label)).long()
    test_label_2 = torch.from_numpy(np.array(test_label_2)).long()


    _val_data = np.array(pd.read_csv(path+psdata_filename[str(27)], header=None))
    _val_label = np.array(pd.read_csv(path+label_filename[str(27)], header=None)).reshape(-1,)
    val_data = []
    val_label = []
    val_label_2 = []
    for i in range(_val_data.shape[0]-20):
        val_data.append(_val_data[i:i+20])
        val_label.append(_val_label[i+19])

    for i in range(len(val_label)):
        if val_label[i] < 2:
            val_label_2.append(0)
        else:
            val_label_2.append(1)

    val_data = torch.from_numpy(np.array(val_data)).to(torch.float32)
    val_label = torch.from_numpy(np.array(val_label)).long()
    val_label_2 = torch.from_numpy(np.array(val_label_2)).long()


    train_index_2 = np.argwhere(train_label<2).squeeze()
    val_index_2 = np.argwhere(val_label<2).squeeze()
    test_index_2 = np.argwhere(test_label<2).squeeze()
    
    
    print(train_data.shape, test_data.shape, val_data.shape)
    
    
    


    # Initialize baseline
    
    baseline = RolloutBaseline(model, val_data, val_label_2, opts)  #oba

    baseline_2 = RolloutBaseline(model_2, val_data[val_index_2], val_label[val_index_2], opts) #oba



    # Initialize optimizer
    optimizer = optim.Adam(
        [{'params': model.parameters(), 'lr': opts.lr_meta_model}]
        + (
            [{'params': baseline.get_learnable_parameters(), 'lr': opts.lr_critic}]
            if len(baseline.get_learnable_parameters()) > 0
            else []
        )
    )
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.001)


    # Initialize learning rate scheduler, decay by lr_decay once per epoch!
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    lr_scheduler_2 = optim.lr_scheduler.StepLR(optimizer_2, step_size=1, gamma=0.95)

    # Start the actual training loop
  
    opts.num_images = train_data.shape[0]
    opts.budget = int(opts.num_images*0.1)
    opts.initial_budget = int(opts.num_images*0.6)
    opts.nc = 20 #OBA
    # vae = AAE(32).to(opts.device)
    aae = AAE(40,128,8,normalization=opts.normalization).to(opts.device) #OBA
    discriminator = Discriminator(128).to(opts.device)
    # optim_vae = optim.Adam(vae.parameters(), lr=5e-4)
    optim_aae = optim.Adam(aae.parameters(), lr=5e-4)
    optim_discriminator = optim.Adam(discriminator.parameters(), lr=5e-4)
    all_indices = set(np.arange(opts.num_images))
    initial_indices = random.sample(list(all_indices), opts.initial_budget)
    sampler = Data.sampler.SubsetRandomSampler(list(initial_indices))
    current_indices = list(initial_indices)
    sampling_indices = []
    unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
    unlabeled_sampler = Data.sampler.SubsetRandomSampler(unlabeled_indices)
    train_dataset = baseline.wrap_dataset(train_data, train_label_2, subject_index) #OBA
    querry_dataloader = Data.DataLoader(train_dataset, sampler=sampler, batch_size=opts.batch_size, num_workers=0)
    solver = Solver(opts)
    splits = [0.6, 0.7, 0.8, 0.9]

    max_test_acc = 0
    acc_list = []
    total_epoch = 0
    if opts.eval_only:
        validate(model, val_data, val_label, opts)
    else:
        for split in splits:
            for epoch in range(10):
                test_acc =  train_epoch(
                                model,
                                optimizer,
                                baseline,
                                lr_scheduler,
                                total_epoch,
                                train_data,
                                train_label_2,  #oba
                                subject_index,
                                test_data, 
                                test_label_2,   #oba
                                tb_logger,
                                opts,
                                sampler
                            )
                max_test_acc = max(test_acc,max_test_acc)
                acc_list.append(test_acc.item())
                # if total_epoch-baseline.epoch>8 and total_epoch>10:
                #     total_epoch += 1
                #     break
                # total_epoch += 1

                print('testing max_test_acc: {}'.format(max_test_acc))
                
                # VAE step
                unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
                unlabeled_sampler = Data.sampler.SubsetRandomSampler(unlabeled_indices)
                unlabeled_dataloader = Data.DataLoader(train_dataset, 
                    sampler=unlabeled_sampler, batch_size=opts.batch_size, drop_last=False)
                
                labeled_data = solver.read_data(querry_dataloader)
                unlabeled_data = solver.read_data(unlabeled_dataloader, labels=True)
                labeled_imgs, labels, index_labeled = next(labeled_data)
                unlabeled_imgs, labels_2, index_unlabeled = next(unlabeled_data)
                label_batch_view_1 = torch.tensor([])
                label_batch_view_2 = torch.tensor([])
                unlabel_batch_view_1 = torch.tensor([])
                unlabel_batch_view_2 = torch.tensor([])
                for i in range(len(index_labeled)-1):
                    bo = 1
                    for j in range(i+1, len(index_labeled)):
                        if index_labeled[i] == index_labeled[j] and labels[i] == labels[j]:
                            label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                            label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                            bo = 0
                            break
                    if bo:
                        label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                for j in range(len(index_labeled)):
                    if index_labeled[-1] == index_labeled[j] and labels[-1] == labels[j]:
                        label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                        label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                        break
                for i in range(len(index_unlabeled)-1):
                    bo = 1
                    for j in range(i+1, len(index_unlabeled)):
                        if index_unlabeled[i] == index_unlabeled[j] and labels_2[i] == labels_2[j]:
                            unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                            unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                            bo = 0
                            break
                    if bo:
                        unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                for j in range(len(index_unlabeled)):
                    if index_unlabeled[-1] == index_unlabeled[j] and labels_2[-1] == labels_2[j]:
                        unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                        unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                        break
                # labeled_imgs = labeled_imgs.to(opts.device)
                # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                label_batch_view_1 = label_batch_view_1.to(opts.device)
                label_batch_view_2 = label_batch_view_2.to(opts.device)
                unlabel_batch_view_1 = unlabel_batch_view_1.to(opts.device)
                unlabel_batch_view_2 = unlabel_batch_view_2.to(opts.device)
                labels = labels.to(opts.device)
                for count in range(opts.num_vae_steps):
                    # recon, z, mu, logvar = vae(labeled_imgs)
                    recon, mu, contr_loss_1 = aae(label_batch_view_1, label_batch_view_2)
                    # unsup_loss = solver.vae_loss(labeled_imgs, recon, mu, logvar, opts.beta)
                    unsup_loss = solver.aae_loss(label_batch_view_1, recon)
                    # unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                    unlab_recon, unlab_mu, contr_loss_2 = aae(unlabel_batch_view_1, unlabel_batch_view_2)
                    contr_loss = contr_loss_1 + contr_loss_2
                    # transductive_loss = solver.vae_loss(unlabeled_imgs, 
                    #         unlab_recon, unlab_mu, unlab_logvar, opts.beta)
                    transductive_loss = solver.aae_loss(unlabel_batch_view_1, unlab_recon)
                
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0),1)
                    unlab_real_preds = torch.ones(unlabeled_imgs.size(0),1)
                        

                    lab_real_preds = lab_real_preds.to(opts.device)
                    unlab_real_preds = unlab_real_preds.to(opts.device)

                    dsc_loss = solver.bce_loss(labeled_preds, lab_real_preds) + \
                            solver.bce_loss(unlabeled_preds, unlab_real_preds)
                    #total_vae_loss = unsup_loss + transductive_loss + opts.adversary_param * dsc_loss
                    #total_vae_loss = unsup_loss + transductive_loss + opts.adversary_param * dsc_loss + contr_loss
                    total_vae_loss = opts.adversary_param * dsc_loss + contr_loss
                    optim_aae.zero_grad()
                    total_vae_loss.backward()
                    optim_aae.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (opts.num_vae_steps - 1):
                        labeled_imgs, _, index_labeled = next(labeled_data)
                        unlabeled_imgs, _, index_unlabeled= next(unlabeled_data)

                        label_batch_view_1 = torch.tensor([])
                        label_batch_view_2 = torch.tensor([])
                        unlabel_batch_view_1 = torch.tensor([])
                        unlabel_batch_view_2 = torch.tensor([])
                        for i in range(len(index_labeled)-1):
                            bo = 1
                            for j in range(i+1, len(index_labeled)):
                                if index_labeled[i] == index_labeled[j] and labels[i] == labels[j]:
                                    label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                    label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                    bo = 0
                                    break
                            if bo:
                                label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        for j in range(len(index_labeled)):
                            if index_labeled[-1] == index_labeled[j] and labels[-1] == labels[j]:
                                label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                                label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                break
                        for i in range(len(index_unlabeled)-1):
                            bo = 1
                            for j in range(i+1, len(index_unlabeled)):
                                if index_unlabeled[i] == index_unlabeled[j] and labels_2[i] == labels_2[j]:
                                    unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                    unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                    bo = 0
                                    break
                            if bo:
                                unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        for j in range(len(index_unlabeled)):
                            if index_unlabeled[-1] == index_unlabeled[j] and labels_2[-1] == labels_2[j]:
                                unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                                unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                break
                        # labeled_imgs = labeled_imgs.to(opts.device)
                        # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                        label_batch_view_1 = label_batch_view_1.to(opts.device)
                        label_batch_view_2 = label_batch_view_2.to(opts.device)
                        unlabel_batch_view_1 = unlabel_batch_view_1.to(opts.device)
                        unlabel_batch_view_2 = unlabel_batch_view_2.to(opts.device)
                        # labeled_imgs = labeled_imgs.to(opts.device)
                        # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                        # labels = labels.to(opts.device)

                # Discriminator step
                for count in range(opts.num_adv_steps):
                    with torch.no_grad():
                        # _, _, mu, _ = vae(labeled_imgs)
                        # _, _, unlab_mu, _ = vae(unlabeled_imgs)
                        _, mu, _= aae(label_batch_view_1, label_batch_view_2)
                        _, unlab_mu, _ = aae(unlabel_batch_view_1, unlabel_batch_view_2)
                    
                    labeled_preds = discriminator(mu)
                    unlabeled_preds = discriminator(unlab_mu)
                    
                    lab_real_preds = torch.ones(labeled_imgs.size(0),1)
                    unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0),1)

                
                    lab_real_preds = lab_real_preds.cuda()
                    unlab_fake_preds = unlab_fake_preds.cuda()
                    
                    dsc_loss = solver.bce_loss(labeled_preds, lab_real_preds) + \
                            solver.bce_loss(unlabeled_preds, unlab_fake_preds)

                    optim_discriminator.zero_grad()
                    dsc_loss.backward()
                    optim_discriminator.step()

                    # sample new batch if needed to train the adversarial network
                    if count < (opts.num_adv_steps - 1):
                        labeled_imgs, _, index_labeled = next(labeled_data)
                        unlabeled_imgs, _, index_unlabeled= next(unlabeled_data)

                        label_batch_view_1 = torch.tensor([])
                        label_batch_view_2 = torch.tensor([])
                        unlabel_batch_view_1 = torch.tensor([])
                        unlabel_batch_view_2 = torch.tensor([])
                        for i in range(len(index_labeled)-1):
                            bo = 1
                            for j in range(i+1, len(index_labeled)):
                                if index_labeled[i] == index_labeled[j] and labels[i] == labels[j]:
                                    label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                    label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                    bo = 0
                                    break
                            if bo:
                                label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        for j in range(len(index_labeled)):
                            if index_labeled[-1] == index_labeled[j] and labels[-1] == labels[j]:
                                label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                                label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                break
                        for i in range(len(index_unlabeled)-1):
                            bo = 1
                            for j in range(i+1, len(index_unlabeled)):
                                if index_unlabeled[i] == index_unlabeled[j] and labels_2[i] == labels_2[j]:
                                    unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                    unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                    bo = 0
                                    break
                            if bo:
                                unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        for j in range(len(index_unlabeled)):
                            if index_unlabeled[-1] == index_unlabeled[j] and labels_2[-1] == labels_2[j]:
                                unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                                unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                break
                        # labeled_imgs = labeled_imgs.to(opts.device)
                        # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                        label_batch_view_1 = label_batch_view_1.to(opts.device)
                        label_batch_view_2 = label_batch_view_2.to(opts.device)
                        unlabel_batch_view_1 = unlabel_batch_view_1.to(opts.device)
                        unlabel_batch_view_2 = unlabel_batch_view_2.to(opts.device)
                        # labeled_imgs = labeled_imgs.to(opts.device)
                        # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                        # labels = labels.to(opts.device)
                print('Current task model acc: {:.4f}'.format(test_acc.item()))
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))
                if total_epoch-baseline.epoch>8 and total_epoch>15:
                    total_epoch += 1
                    break
                total_epoch += 1
                # if split>0.4 and epoch-baseline.epoch>2:
                #     break
            print('Final accuracy with {}% of data is: {:.5f}'.format(int(split*100), test_acc))
            
            sampled_indices = solver.sample_for_labeling(aae, discriminator, unlabeled_dataloader)
            current_indices = list(current_indices) + list(sampled_indices)
            sampling_indices = list(sampling_indices) + list(sampled_indices)
            sampler = Data.sampler.SubsetRandomSampler(current_indices)
            querry_dataloader = Data.DataLoader(train_dataset, sampler=sampler, batch_size=opts.batch_size, num_workers=0)
            
    # sampling_indices = np.array(sampling_indices)
    # np.savetxt('data/OBA_4_sampling_indices.txt', sampling_indices)
    test_label = test_label.to(opts.device)
    test_label_2 = test_label_2.to(opts.device) #oba

    pred, test_embed = LOO(model, test_data, test_label_2, opts) #oba
    acc = torch.eq(pred.max(1)[1], test_label_2).float().mean() #oba
    print('testing acc: {}'.format(acc))

   
    
    train_data_3 = train_data[train_index_2]  
    train_label_3 = train_label[train_index_2]
    subject_index_3 = subject_index[train_index_2]
    # opts.alpha = 0.7
    opts.batch_size = 250
    max_test_acc = 0
    opts.num_images = train_data_3.shape[0]
    opts.budget = int(opts.num_images*0.1)
    opts.initial_budget = int(opts.num_images*0.5)
    aae_2 = AAE(40,128,8,normalization=opts.normalization).to(opts.device)
    discriminator_2 = Discriminator(128).to(opts.device)
    optim_aae_2 = optim.Adam(aae_2.parameters(), lr=5e-4)
    optim_discriminator_2 = optim.Adam(discriminator_2.parameters(), lr=5e-4)
    all_indices = set(np.arange(opts.num_images))
    initial_indices = random.sample(list(all_indices), opts.initial_budget)
    sampler = Data.sampler.SubsetRandomSampler(list(initial_indices))
    current_indices = list(initial_indices)
    train_dataset = baseline.wrap_dataset(train_data_3, train_label_3, subject_index_3)
    querry_dataloader = Data.DataLoader(train_dataset, sampler=sampler, batch_size=opts.batch_size, num_workers=0)
    solver1 = Solver(opts)
    splits = [0.5, 0.6,0.7,0.8,0.9]
    total_epoch = 0
    # for epoch in range(100):
    #         test_acc =  train_epoch(
    #                         model_2,
    #                         optimizer_2,
    #                         baseline_2,
    #                         lr_scheduler_2,
    #                         epoch,
    #                         train_data_3,
    #                         train_label_3,
    #                         test_data[test_index_2],
    #                         test_label[test_index_2],
    #                         tb_logger,
    #                         opts,
    #                         sampler
    #                     )
    #         max_test_acc = max(test_acc,max_test_acc)
    #         print('testing max_test_acc: {}'.format(max_test_acc))
    for split in splits:
        for epoch in range(10):
            test_acc =  train_epoch(
                        model_2,
                        optimizer_2,
                        baseline_2,
                        lr_scheduler_2,
                        total_epoch,
                        train_data_3,
                        train_label_3,
                        subject_index_3,
                        test_data[test_index_2],
                        test_label[test_index_2],
                        tb_logger,
                        opts,
                        sampler
                    )
            max_test_acc = max(test_acc,max_test_acc)
            # if total_epoch-baseline_2.epoch>8 and epoch>3:
            #     total_epoch += 1
            #     break
            # total_epoch += 1
            print('testing max_test_acc: {}'.format(max_test_acc))
            
            #total_epoch += 1
            
            # VAE step
            unlabeled_indices = np.setdiff1d(list(all_indices), current_indices)
            unlabeled_sampler = Data.sampler.SubsetRandomSampler(unlabeled_indices)
            unlabeled_dataloader = Data.DataLoader(train_dataset, 
                sampler=unlabeled_sampler, batch_size=opts.batch_size, drop_last=False)
            
            labeled_data = solver.read_data(querry_dataloader)
            unlabeled_data = solver.read_data(unlabeled_dataloader, labels=True)
            labeled_imgs, labels, index_labeled = next(labeled_data)
            unlabeled_imgs, labels_2, index_unlabeled = next(unlabeled_data)
            label_batch_view_1 = torch.tensor([])
            label_batch_view_2 = torch.tensor([])
            unlabel_batch_view_1 = torch.tensor([])
            unlabel_batch_view_2 = torch.tensor([])
            for i in range(len(index_labeled)-1):
                bo = 1
                for j in range(i+1, len(index_labeled)):
                    if index_labeled[i] == index_labeled[j] and labels[i] == labels[j]:
                        label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                        bo = 0
                        break
                if bo:
                    label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                    label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
            for j in range(len(index_labeled)):
                if index_labeled[-1] == index_labeled[j] and labels[-1] == labels[j]:
                    label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                    label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                    break
            for i in range(len(index_unlabeled)-1):
                bo = 1
                for j in range(i+1, len(index_unlabeled)):
                    if index_unlabeled[i] == index_unlabeled[j] and labels_2[i] == labels_2[j]:
                        unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                        unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                        bo = 0
                        break
                if bo:
                    unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                    unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
            for j in range(len(index_unlabeled)):
                if index_unlabeled[-1] == index_unlabeled[j] and labels_2[-1] == labels_2[j]:
                    unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                    unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                    break
            # labeled_imgs = labeled_imgs.to(opts.device)
            # unlabeled_imgs = unlabeled_imgs.to(opts.device)
            label_batch_view_1 = label_batch_view_1.to(opts.device)
            label_batch_view_2 = label_batch_view_2.to(opts.device)
            unlabel_batch_view_1 = unlabel_batch_view_1.to(opts.device)
            unlabel_batch_view_2 = unlabel_batch_view_2.to(opts.device)
            labels = labels.to(opts.device)
            for count in range(opts.num_vae_steps):
                # recon, z, mu, logvar = vae(labeled_imgs)
                recon, mu, contr_loss_1 = aae_2(label_batch_view_1, label_batch_view_2)
                # unsup_loss = solver.vae_loss(labeled_imgs, recon, mu, logvar, opts.beta)
                unsup_loss = solver.aae_loss(label_batch_view_1, recon)
                # unlab_recon, unlab_z, unlab_mu, unlab_logvar = vae(unlabeled_imgs)
                unlab_recon, unlab_mu, contr_loss_2 = aae_2(unlabel_batch_view_1, unlabel_batch_view_2)
                contr_loss = contr_loss_1 + contr_loss_2
                # transductive_loss = solver.vae_loss(unlabeled_imgs, 
                #         unlab_recon, unlab_mu, unlab_logvar, opts.beta)
                transductive_loss = solver.aae_loss(unlabel_batch_view_1, unlab_recon)
            
                labeled_preds = discriminator_2(mu)
                unlabeled_preds = discriminator_2(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0),1)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0),1)
                    

                lab_real_preds = lab_real_preds.to(opts.device)
                unlab_real_preds = unlab_real_preds.to(opts.device)

                dsc_loss = solver.bce_loss(labeled_preds, lab_real_preds) + \
                        solver.bce_loss(unlabeled_preds, unlab_real_preds)
                #total_vae_loss = unsup_loss + transductive_loss + opts.adversary_param * dsc_loss
                #total_vae_loss = unsup_loss + transductive_loss + opts.adversary_param * dsc_loss + contr_loss
                total_vae_loss = opts.adversary_param * dsc_loss + contr_loss
                optim_aae_2.zero_grad()
                total_vae_loss.backward()
                optim_aae_2.step()

                # sample new batch if needed to train the adversarial network
                if count < (opts.num_vae_steps - 1):
                    labeled_imgs, _, index_labeled = next(labeled_data)
                    unlabeled_imgs, _, index_unlabeled= next(unlabeled_data)

                    label_batch_view_1 = torch.tensor([])
                    label_batch_view_2 = torch.tensor([])
                    unlabel_batch_view_1 = torch.tensor([])
                    unlabel_batch_view_2 = torch.tensor([])
                    for i in range(len(index_labeled)-1):
                        bo = 1
                        for j in range(i+1, len(index_labeled)):
                            if index_labeled[i] == index_labeled[j] and labels[i] == labels[j]:
                                label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                bo = 0
                                break
                        if bo:
                            label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                            label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                    for j in range(len(index_labeled)):
                        if index_labeled[-1] == index_labeled[j] and labels[-1] == labels[j]:
                            label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                            label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                            break
                    for i in range(len(index_unlabeled)-1):
                        bo = 1
                        for j in range(i+1, len(index_unlabeled)):
                            if index_unlabeled[i] == index_unlabeled[j] and labels_2[i] == labels_2[j]:
                                unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                bo = 0
                                break
                        if bo:
                            unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                            unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                    for j in range(len(index_unlabeled)):
                        if index_unlabeled[-1] == index_unlabeled[j] and labels_2[-1] == labels_2[j]:
                            unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                            unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                            break
                    # labeled_imgs = labeled_imgs.to(opts.device)
                    # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                    label_batch_view_1 = label_batch_view_1.to(opts.device)
                    label_batch_view_2 = label_batch_view_2.to(opts.device)
                    unlabel_batch_view_1 = unlabel_batch_view_1.to(opts.device)
                    unlabel_batch_view_2 = unlabel_batch_view_2.to(opts.device)
                    # labeled_imgs = labeled_imgs.to(opts.device)
                    # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                    # labels = labels.to(opts.device)

            # Discriminator step
            for count in range(opts.num_adv_steps):
                with torch.no_grad():
                    # _, _, mu, _ = vae(labeled_imgs)
                    # _, _, unlab_mu, _ = vae(unlabeled_imgs)
                    _, mu, _= aae(label_batch_view_1, label_batch_view_2)
                    _, unlab_mu, _ = aae(unlabel_batch_view_1, unlabel_batch_view_2)
                
                labeled_preds = discriminator(mu)
                unlabeled_preds = discriminator(unlab_mu)
                
                lab_real_preds = torch.ones(labeled_imgs.size(0),1)
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0),1)

            
                lab_real_preds = lab_real_preds.cuda()
                unlab_fake_preds = unlab_fake_preds.cuda()
                
                dsc_loss = solver.bce_loss(labeled_preds, lab_real_preds) + \
                        solver.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator_2.zero_grad()
                dsc_loss.backward()
                optim_discriminator_2.step()

                # sample new batch if needed to train the adversarial network
                if count < (opts.num_adv_steps - 1):
                    labeled_imgs, _, index_labeled = next(labeled_data)
                    unlabeled_imgs, _, index_unlabeled= next(unlabeled_data)

                    label_batch_view_1 = torch.tensor([])
                    label_batch_view_2 = torch.tensor([])
                    unlabel_batch_view_1 = torch.tensor([])
                    unlabel_batch_view_2 = torch.tensor([])
                    for i in range(len(index_labeled)-1):
                        bo = 1
                        for j in range(i+1, len(index_labeled)):
                            if index_labeled[i] == index_labeled[j] and labels[i] == labels[j]:
                                label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                bo = 0
                                break
                        if bo:
                            label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                            label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                    for j in range(len(index_labeled)):
                        if index_labeled[-1] == index_labeled[j] and labels[-1] == labels[j]:
                            label_batch_view_1 = torch.cat([label_batch_view_1, labeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                            label_batch_view_2 = torch.cat([label_batch_view_2, labeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                            break
                    for i in range(len(index_unlabeled)-1):
                        bo = 1
                        for j in range(i+1, len(index_unlabeled)):
                            if index_unlabeled[i] == index_unlabeled[j] and labels_2[i] == labels_2[j]:
                                unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                                unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                                bo = 0
                                break
                        if bo:
                            unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                            unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[i].reshape(1,opts.nc,-1)], 0)
                    for j in range(len(index_unlabeled)):
                        if index_unlabeled[-1] == index_unlabeled[j] and labels_2[-1] == labels_2[j]:
                            unlabel_batch_view_1 = torch.cat([unlabel_batch_view_1, unlabeled_imgs[-1].reshape(1,opts.nc,-1)], 0)
                            unlabel_batch_view_2 = torch.cat([unlabel_batch_view_2, unlabeled_imgs[j].reshape(1,opts.nc,-1)], 0)
                            break
                    # labeled_imgs = labeled_imgs.to(opts.device)
                    # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                    label_batch_view_1 = label_batch_view_1.to(opts.device)
                    label_batch_view_2 = label_batch_view_2.to(opts.device)
                    unlabel_batch_view_1 = unlabel_batch_view_1.to(opts.device)
                    unlabel_batch_view_2 = unlabel_batch_view_2.to(opts.device)
                    # labeled_imgs = labeled_imgs.to(opts.device)
                    # unlabeled_imgs = unlabeled_imgs.to(opts.device)
                    # labels = labels.to(opts.device)
            print('Current task model acc: {:.4f}'.format(test_acc.item()))
            print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
            print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))
            if total_epoch-baseline_2.epoch>8 and epoch>3:
                total_epoch += 1
                break
            total_epoch += 1
            # if split>0.4 and epoch-baseline.epoch>3:
            #     break
        print('Final accuracy with {}% of data is: {:.5f}'.format(int(split*100), test_acc))

        sampled_indices = solver1.sample_for_labeling(aae_2, discriminator_2, unlabeled_dataloader)
        current_indices = list(current_indices) + list(sampled_indices)
        sampler = Data.sampler.SubsetRandomSampler(current_indices)
        querry_dataloader = Data.DataLoader(train_dataset, sampler=sampler, batch_size=opts.batch_size, num_workers=0)
        
    
    pred_result = pred.max(1)[1]
    pred_index = []
    for i in range(len(pred)):
        if pred_result[i] == 0:
            pred_index.append(i)
        else:
            pred_result[i] = 2
    pred_index = np.array(pred_index)
    new_pred, test_embed = LOO(model_2, test_data[pred_index], test_label[pred_index], opts)

    new_pred_result = new_pred.max(1)[1]
    new_acc = torch.eq(new_pred_result, test_label[pred_index]).float().mean()
    print('01 testing acc: {}'.format(new_acc))
    for i in range(len(pred_index)):
        pred_result[pred_index[i]] = new_pred_result[i]

    
    acc = torch.eq(pred_result, test_label).float().mean()
    print('final testing acc: {}'.format(acc))
    
    # labels_name = ['0','1','2']

    # cm = confusion_matrix(pred_result.cpu().numpy(), test_label.cpu().numpy())
    # print(cm)
    # plot_confusion_matrix(cm, labels_name, "MSAE Confusion Matrix")
    # plt.show()
    



            


if __name__ == "__main__":
    run(get_options())
