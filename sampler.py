import torch

import numpy as np

class AdversarySampler:
    def __init__(self, budget):
        self.budget = budget


    # def sample(self, vae, discriminator, data, cuda):
    #     all_preds = []
    #     all_indices = []
        
    #     for batch in data:
    #         images, _, _, indices = batch['data'], batch['label'], batch['baseline'], batch['index']
    #     # for images, _, indices in data:
    #         if cuda:
    #             images = images.cuda()

    #         with torch.no_grad():
    #             _, _, mu, _ = vae(images)
    #             preds = discriminator(mu)

    #         preds = preds.cpu().data
    #         all_preds.extend(preds)
    #         all_indices.extend(indices)

    #     all_preds = torch.stack(all_preds)
    #     all_preds = all_preds.view(-1)
    #     # need to multiply by -1 to be able to use torch.topk 
    #     all_preds *= -1

    #     # select the points which the discriminator things are the most likely to be unlabeled
    #     _, querry_indices = torch.topk(all_preds, int(self.budget))
    #     querry_pool_indices = np.asarray(all_indices)[querry_indices]

    #     return querry_pool_indices

    def sample(self, aae, discriminator, data, cuda, nc = 20):
        all_preds = []
        all_indices = []
        
        
        for batch in data:
            images, labels, index, indices = batch['data'], batch['label'], batch['subject_index'], batch['index']
        # for images, _, indices in data:
            batch_view_1 = torch.tensor([])
            batch_view_2 = torch.tensor([])
            for i in range(len(index)-1):
                bo = 1
                for j in range(i+1, len(index)):
                    if index[i] == index[j] and labels[i] == labels[j]:
                        batch_view_1 = torch.cat([batch_view_1, images[i].reshape(1,nc,-1)], 0)
                        batch_view_2 = torch.cat([batch_view_2, images[j].reshape(1,nc,-1)], 0)
                        bo = 0
                        break
                if bo:
                    batch_view_1 = torch.cat([batch_view_1, images[i].reshape(1,nc,-1)], 0)
                    batch_view_2 = torch.cat([batch_view_2, images[i].reshape(1,nc,-1)], 0)
            for j in range(len(index)):
                if index[-1] == index[j] and labels[-1] == labels[j]:
                    batch_view_1 = torch.cat([batch_view_1, images[-1].reshape(1,nc,-1)], 0)
                    batch_view_2 = torch.cat([batch_view_2, images[j].reshape(1,nc,-1)], 0)
                    break
            if cuda:
                batch_view_1 = batch_view_1.cuda()
                batch_view_2 = batch_view_2.cuda()

            with torch.no_grad():
                _, mu, _ = aae(batch_view_1, batch_view_2)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk 
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices
        
