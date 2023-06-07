from copy import deepcopy
import os
import time
from tqdm import tqdm
import torch
import math

from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torch.nn as nn

from utils.log_utils import log_values
from utils import move_to

from torch.autograd import grad

import collections
from collections import Counter
from sklearn.model_selection import train_test_split

import numpy as np

import random
import torch.nn.functional as F


def LOO(model, x, y, opts):

    # Put in greedy evaluation mode!
    x = move_to(x, opts.device)
    y = move_to(y, opts.device)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    # spt_index = []
    # qry_index = []
    # for i in range(len(label)):
    #     if i%2 == 0:
    #         spt_index.append(i)
    #     else:
    #         qry_index.append(i)
    # spt_index = move_to(torch.from_numpy(np.array(spt_index)), opts.device)
    # qry_index = move_to(torch.from_numpy(np.array(qry_index)), opts.device)
    out_result = move_to(torch.tensor([]), opts.device)
    embeddings = move_to(torch.tensor([]), opts.device)
    for batch_id, data in enumerate(DataLoader(x, batch_size=opts.eval_batch_size)):
        label = y[batch_id*opts.eval_batch_size:(batch_id+1)*opts.eval_batch_size]
         
        fast_weights = collections.OrderedDict(model.named_parameters())
            
        result, embed, attn = model.functional_forward(data, fast_weights)

        out_result = torch.cat([out_result, result], 0)
        embeddings = torch.cat([embeddings, embed], 0)
    return out_result, embeddings


def eval(model, x, y, opts):

    # Put in greedy evaluation mode!
    x = move_to(x, opts.device)
    y = move_to(y, opts.device)
    loss_func = nn.CrossEntropyLoss(reduction='none')
    # spt_index = []
    # qry_index = []
    # for i in range(len(label)):
    #     if i%2 == 0:
    #         spt_index.append(i)
    #     else:
    #         qry_index.append(i)
    # spt_index = move_to(torch.from_numpy(np.array(spt_index)), opts.device)
    # qry_index = move_to(torch.from_numpy(np.array(qry_index)), opts.device)
    out_result = move_to(torch.tensor([]), opts.device)
    for batch_id, data in enumerate(DataLoader(x, batch_size=opts.eval_batch_size)):
        label = y[batch_id*opts.eval_batch_size:(batch_id+1)*opts.eval_batch_size]
        index = np.asarray([i for i in range(data.shape[0])])
        np.random.shuffle(index)
        spt_index, qry_index = train_test_split(index,train_size=0.2,random_state=1234)
        

        spt_index = move_to(torch.tensor(spt_index), opts.device)
        qry_index = move_to(torch.tensor(qry_index), opts.device)

        x_spt = torch.index_select(data, 0, spt_index)
        x_qry = torch.index_select(data, 0, qry_index)
        y_spt = torch.index_select(label, 0, spt_index)
        y_qry = torch.index_select(label, 0, qry_index)
        
        fast_weights = collections.OrderedDict(model.named_parameters())
        for i in range(opts.support_inner_loop):

            # Evaluate model, get costs and log probabilities
            outputs, embed = model.functional_forward(x_qry[:x_spt.shape[0]], fast_weights)

            loss = loss_func(outputs, y_qry[:x_spt.shape[0]]).mean()

            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                    for ((name, param), grads) in zip(fast_weights.items(), grads))
                
        # with torch.no_grad():
        spt_outputs, embed = model.functional_forward(x_spt, fast_weights)


        fast_weights = collections.OrderedDict(model.named_parameters())
        for i in range(opts.support_inner_loop):

            # Evaluate model, get costs and log probabilities
            outputs, embed = model.functional_forward(x_spt, fast_weights)

            loss = loss_func(outputs, y_spt).mean()

            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                    for ((name, param), grads) in zip(fast_weights.items(), grads))
            
        qry_outputs, embed = model.functional_forward(x_qry, fast_weights)
        result = move_to(torch.zeros(len(label),3), opts.device)
        for i in range(len(spt_index)):
            result[spt_index[i]] = spt_outputs[i]
        for i in range(len(qry_index)):
            result[qry_index[i]] = qry_outputs[i]
        out_result = torch.cat([out_result, result], 0)
    return out_result
    
def regression_loss(x, y):
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return 2 - 2 * (x * y).sum(dim=-1)


def  get_inner_model(model):
    return model.module if isinstance(model, DataParallel) else model


def validate(model, data, label, baseline, opts):
    # Validate
    print('testing...')
    # data = baseline.wrap_dataset(data, label)

    test_acc, bl_val = rollout(model, data, label, opts)

    print('testing overall acc: {} '.format(
        test_acc))

    return test_acc


def rollout(model, data, label, opts):
    # Put in greedy evaluation mode!
    fast_weights = collections.OrderedDict(model.named_parameters())
    loss_func = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    out = move_to(torch.tensor([]), opts.device)
    label = move_to(label, opts.device)
    def eval_model_bat(bat):
        with torch.no_grad():
            outputs, embed, attn = model.functional_forward(move_to(bat, opts.device), fast_weights)
        return outputs
    
    for bat in DataLoader(data, batch_size=opts.eval_batch_size):
        out = torch.cat([out, eval_model_bat(bat)], 0)
    
    acc = torch.eq(out.max(1)[1], label).float().mean()
    _bl = out.softmax(dim=1)
    bl = _bl.gather(1, label.view(-1,1))
    return acc.data.cpu(), bl.data.cpu()


def rollout_eval(model, data, label, bl_val, opts):

    # Put in greedy evaluation mode!

    def eval_model_bat(x, label, bl_val):
        x = move_to(x, opts.device)
        label = move_to(label, opts.device)
        bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
        # bl_val = None
        # bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
        loss_func = nn.CrossEntropyLoss(reduction='none')

        ''''''

        update_step = 1
        # losses_q = [0 for _ in range(update_step + 2)]  # losses_q[i] is the loss on step i
        # costs = [0 for _ in range(update_step + 2)]
        
        index = np.asarray([i for i in range(x.shape[0])])
        np.random.shuffle(index)
        x_spt_index, x_qry_index = train_test_split(index,train_size=0.2,random_state=1234)
        

        x_spt_index = move_to(torch.tensor(x_spt_index), opts.device)
        x_qry_index = move_to(torch.tensor(x_qry_index), opts.device)
        x_spt = torch.index_select(x, 0, x_spt_index)
        x_qry = torch.index_select(x, 0, x_qry_index)
        y_spt = torch.index_select(label, 0, x_spt_index)
        y_qry = torch.index_select(label, 0, x_qry_index)

        
        for i in range(opts.support_inner_loop):
            fast_weights = collections.OrderedDict(model.named_parameters())

            # Evaluate model, get costs and log probabilities
            outputs, embed = model.functional_forward(x_spt, fast_weights)

            if bl_val is None:
                bl_acc, bl_val = rollout(model, x_spt, opts)
                bl_val_spt = move_to(bl_val, opts.device)
            else:
                bl_val_spt = torch.index_select(bl_val, 0, x_spt_index)

            # Evaluate baseline, get baseline loss if any (only for critic)
            # bl_val, bl_loss = baseline.eval(x_spt, cost) if bl_val is None else (bl_val, 0)
            acc_spt = torch.eq(outputs.max(1)[1], y_spt).float().mean()
            # Calculate loss
            # loss = loss_func(outputs, y_spt)*(acc_spt - bl_val) 
            loss = (loss_func(outputs, y_spt)).mean()

            # bl_loss.requires_grad_()
            # loss = reinforce_loss + bl_loss

            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                    for ((name, param), grads) in zip(fast_weights.items(), grads))
                
        # with torch.no_grad():
        outputs, embed = model.functional_forward(x_qry, fast_weights)
        acc = torch.eq(outputs.max(1)[1], y_qry).float().mean()

        return acc.data.cpu()

    # true_label = move_to(torch.tensor([]), opts.device)
    # losses = []
    acc = 0
    if bl_val is not None:
        baseline = 1
    else:
        baseline = 0
    for batch_id, batch in enumerate(DataLoader(data, batch_size=opts.eval_batch_size)):
        
        if baseline:
            bat = batch
            bl_val_in = bl_val[batch_id*opts.eval_batch_size:(batch_id+1)*opts.eval_batch_size]
        else:
            bat, bl_val_in = batch['data'], batch['baseline']
        pred_acc = eval_model_bat(bat, label[batch_id*opts.eval_batch_size:(batch_id+1)*opts.eval_batch_size], bl_val_in)
        acc += pred_acc
        # outputs = torch.cat([outputs, pred],0)
        # true_label = torch.cat([true_label, true], 0)
    
    acc = acc / (batch_id+1)
    return acc.data.cpu()

def clip_grad_norms(param_groups, max_norm=math.inf):
    """
    Clips the norms for all param groups to max_norm and returns gradient norms before clipping
    :param optimizer:
    :param max_norm:
    :param gradient_norms_log:
    :return: grad_norms, clipped_grad_norms: list with (clipped) gradient norms per group
    """
    grad_norms = [
        torch.nn.utils.clip_grad_norm_(
            group['params'],
            max_norm if max_norm > 0 else math.inf,  # Inf so no clipping but still call to calc
            norm_type=2
        )
        for group in param_groups
    ]
    grad_norms_clipped = [min(g_norm, max_norm) for g_norm in grad_norms] if max_norm > 0 else grad_norms
    return grad_norms, grad_norms_clipped


def train_epoch(model, optimizer, baseline, lr_scheduler, epoch, train_data, train_label, subject_index, test_data, test_label,  tb_logger, opts, sampler):
    print("Start train epoch {}, lr={} for run {}".format(epoch, optimizer.param_groups[0]['lr'], opts.run_name))
    start_time = time.time()


    # Generate new training data for each epoch
    training_dataset = baseline.wrap_dataset(train_data, train_label, subject_index)
    training_dataloader = DataLoader(training_dataset, sampler=sampler, batch_size=opts.batch_size, num_workers=0)

    # Put model in train mode!
    model.train()
    
    #baseline_list里降序存储baseline,取最小的后3个和随机的2个baseline
    
    # if len(baseline.baseline_list) < 5:
    #     baseline.baseline = baseline.baseline_list[random.randint(0, len(baseline.baseline_list)-1)]
    # else:
    #     baseline.baseline = baseline.baseline_list[random.randint(len(baseline.baseline_list)-5, len(baseline.baseline_list)-1)]
    
    for batch_id, batch in enumerate(tqdm(training_dataloader, disable=opts.no_progress_bar)):
        train_batch(
            model,
            optimizer,
            baseline,
            epoch,
            batch_id,
            batch,
            tb_logger,
            opts
        )


    epoch_duration = time.time() - start_time
    print("Finished epoch {}, took {} s".format(epoch, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

    # if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
    #     print('Saving model and state...')
    #     torch.save(
    #         {
    #             'model': get_inner_model(model).state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'rng_state': torch.get_rng_state(),
    #             'cuda_rng_state': torch.cuda.get_rng_state_all(),
    #             'baseline': baseline.state_dict()
    #         },
    #         os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
    #     )

    test_acc = validate(model, test_data, test_label, baseline, opts)

    # if not opts.no_tensorboard:
    #     tb_logger.log_value('val_avg_reward', val_loss, step)
    
    baseline.epoch_callback(model, epoch)

    # lr_scheduler should be called at end of epoch
    lr_scheduler.step()
    return test_acc


def train_batch(
        model,
        optimizer,
        baseline,
        epoch,
        batch_id,
        batch,
        tb_logger,
        opts
):
    x, label, bl_val = baseline.unwrap_batch(batch)
    x = move_to(x, opts.device)
    label = move_to(label, opts.device)
    bl_val = move_to(bl_val, opts.device) if bl_val is not None else None
        
    
    ''''''

    losses_q = [0 for _ in range(opts.query_inner_loop)]  # losses_q[i] is the loss on step i
    train_acc = [0 for _ in range(opts.query_inner_loop)]



    index = np.asarray([i for i in range(x.shape[0])])
    np.random.shuffle(index)
    index = index.reshape(opts.task_num,-1)
    
    loss_func = nn.CrossEntropyLoss(reduction='none')
    
    #随机Sample图中的1/2个点
    for i in range(opts.task_num):
        
        x_spt_index, x_qry_index = train_test_split(index[i],train_size=0.5,random_state=1234)
        x_spt_index = move_to(torch.tensor(x_spt_index), opts.device)
        x_qry_index = move_to(torch.tensor(x_qry_index), opts.device)
        x_spt = torch.index_select(x, 0, x_spt_index)
        x_qry = torch.index_select(x, 0, x_qry_index)
        y_spt = torch.index_select(label, 0, x_spt_index)
        y_qry = torch.index_select(label, 0, x_qry_index)


        fast_weights = collections.OrderedDict(model.named_parameters())
        for j in range(opts.support_inner_loop):
            

            # Evaluate model, get costs and log probabilities
            outputs, embed, attn = model.functional_forward(x_spt, fast_weights)

            acc_spt = outputs.softmax(dim=1)
            acc_spt = acc_spt.gather(1, y_spt.view(-1,1))

            # Evaluate baseline, get baseline loss if any (only for critic)
            bl_val = baseline.eval(x_spt, y_spt) if bl_val is None else bl_val
            # bl_val, bl_loss = baseline.eval(x_spt, cost)
            bl_val_spt = torch.index_select(bl_val, 0, x_spt_index)
            bl_val_qry = torch.index_select(bl_val, 0, x_qry_index)
            
            if epoch < 1:
                loss = loss_func(outputs, y_spt).mean()
            else:
                loss = ((acc_spt-bl_val_spt*opts.alpha)*loss_func(outputs, y_spt)).mean()

            
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                        for ((name, param), grads) in zip(fast_weights.items(), grads))
        

        

        for k in range(opts.query_inner_loop):
            outputs, embed, attn = model.functional_forward(x_qry, fast_weights)
            # con_loss = 0
            # num = 0
            # for q in range(len(y_qry)-1):
            #     for p in range(q+1,len(y_qry)):
            #         if y_qry[p] == y_qry[q] and num<=200:
            #             con_loss += regression_loss(embed[p].view(1,-1), embed[q].view(1,-1))
            #             num += 1

            # contrastive_loss = con_loss/num
            acc_qry = outputs.softmax(dim=1)
            acc_qry = acc_qry.gather(1, y_qry.view(-1,1))

            # Evaluate baseline, get baseline loss if any (only for critic)
            # bl_val, bl_loss = baseline.eval(x, cost) if bl_val is None else (bl_val, 0)
            # bl_val = baseline.meta_eval(x_spt, y_spt, x_qry, y_qry, opts) if bl_val is None else bl_val
            
            if epoch < 1000:
                loss = loss_func(outputs, y_qry).mean()
            else:
                loss = ((bl_val_qry-acc_qry)*loss_func(outputs, y_qry)).mean()

            losses_q[k] += loss
            train_acc[k] += torch.eq(outputs.max(1)[1], y_qry).float().mean()
  
            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                for ((name, param), grads) in zip(fast_weights.items(), grads))
            


        # for p in model.parameters():
        #     p.data = fast_weights[0]
        # get query_set loss
        # Evaluate model, get costs and log probabilities


        
        # optimizer.zero_grad()
        # loss.requires_grad_()
        # loss.backward()
        # grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        # optimizer.step()
        '''
        fast_weights = collections.OrderedDict(model.named_parameters())
        outputs, embed, attn = model.functional_forward(x, fast_weights)

        acc_qry = outputs.softmax(dim=1)
        acc_qry = acc_qry.gather(1, label.view(-1,1))

        
        if epoch < 1000:
            loss = loss_func(outputs, label).mean()
        else:
            loss = ((bl_val-acc_qry)*loss_func(outputs, label)).mean()

        acc = torch.eq(outputs.max(1)[1], label).float().mean()
        '''
        #     optimizer.zero_grad()
        #     loss.requires_grad_()
        #     loss.backward()
        #     grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
        #     optimizer.step()
        
    loss = (losses_q[-1]) / opts.task_num
    acc = train_acc[-1] / opts.task_num


    # grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
    # fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
    #                                                for ((name, param), grads) in zip(fast_weights.items(), grads))

    # fast_weights = list(map(lambda p: p[1] - opts.lr_model * p[0], zip(loss_grad, parameters)))

    # for p in model.parameters():
    #     p.data = fast_weights[0]

    optimizer.zero_grad()
    # loss.requires_grad_()
    loss.backward()
    grad_norms = clip_grad_norms(optimizer.param_groups, opts.max_grad_norm)
    optimizer.step()





