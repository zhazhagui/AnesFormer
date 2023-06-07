import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.stats import ttest_rel
import copy
from meta_train import rollout_eval, get_inner_model, rollout

import collections
import torch.nn as nn

class Baseline(object):

    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass




class RolloutBaseline(Baseline):

    def __init__(self, model, data, label, opts, epoch=0):
        super(Baseline, self).__init__()

        self.opts = opts
        self.data = data
        self.label = label

        self._update_model(model, epoch)

    def _update_model(self, model, epoch):
        self.model = copy.deepcopy(model)
        
        print("Evaluating baseline model on evaluation dataset")
        self.bl_acc, bl_val = rollout(model, self.data, self.label, self.opts)
        # self.bl_acc = rollout_eval(self.model, self.data, self.label, bl_val, self.opts)
        self.epoch = epoch
        if epoch == 0:
            self.bl_acc = torch.tensor(0)

    def wrap_dataset(self, data, label, index):
        print("Evaluating baseline on dataset...")
        # Need to convert baseline to 2D to prevent converting to double, see
        # https://discuss.pytorch.org/t/dataloader-gives-double-instead-of-float/717/3
        bl_acc, bl_val = rollout(self.model, data, label, self.opts)
        return BaselineDataset(data, label, index, bl_val)

    def unwrap_batch(self, batch):

        return batch['data'], batch['label'], batch['baseline']  # Flatten result to undo wrapping as 2D

    def eval(self, x, label):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        fast_weights = collections.OrderedDict(self.model.named_parameters())
        loss_func = nn.CrossEntropyLoss(reduction='none')
        with torch.no_grad():
            v = self.model.functional_forward(x, fast_weights)
            loss = loss_func(v, label)

        # There is no loss
        return loss.data.cpu()
    
    def meta_eval(self, x_spt, y_spt, x_qry, y_qry, opts):
        # Use volatile mode for efficient inference (single batch so we do not use rollout function)
        fast_weights = collections.OrderedDict(self.model.named_parameters())
        loss_func = nn.CrossEntropyLoss(reduction='none')
        optputs = self.model.functional_forward(x_spt, fast_weights)
        loss = loss_func(optputs, y_spt).mean()


        grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
        fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                for ((name, param), grads) in zip(fast_weights.items(), grads))
        
        for i in range(opts.inner_loop):
            optputs = self.model.functional_forward(x_qry, fast_weights)
            loss = loss_func(optputs, y_qry)

            grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            fast_weights = collections.OrderedDict((name, param - opts.lr_model * grads)
                                                for ((name, param), grads) in zip(fast_weights.items(), grads))
        # There is no loss
        return loss.data.cpu()

    def epoch_callback(self, model, epoch):
        """
        Challenges the current baseline with the model and replaces the baseline model if it is improved.
        :param model: The model to challenge the baseline by
        :param epoch: The current epoch
        """
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals_acc, bl_val = rollout(model, self.data, self.label, self.opts)
        # candidate_vals_acc = rollout(model, self.data, self.label, self.opts)
        print("Epoch {} candidate val_acc {}, baseline epoch {} ".format(
            epoch, candidate_vals_acc.item(), self.epoch))
        if candidate_vals_acc.item() > self.bl_acc.item()*1.005:
            print('Update baseline')
            self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'data': self.data,
            'label': self.label,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        # We make it such that it works whether model was saved as data parallel or not
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])


class BaselineDataset(Dataset):

    def __init__(self, dataset=None, label=None, index=None, baseline=None):
        super(BaselineDataset, self).__init__()

        self.dataset = dataset
        self.label = label
        self.baseline = baseline
        self.subject_index = index
         # assert (len(self.dataset) == len(self.baseline))

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'label': self.label[item],
            'subject_index': self.subject_index[item],
            'baseline': self.baseline[item],
            'index': item
        }

    def __len__(self):
        return len(self.dataset)
