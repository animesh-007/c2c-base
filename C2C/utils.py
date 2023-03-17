import torch
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_ckp(state, fpath=None):
    ''' Save model
    '''
    if fpath == None:
        fpath =  'checkpoint.pt'
    torch.save(state, fpath)
    
def load_ckp(checkpoint_fpath, model, optimizer):
    ''' load model
    '''
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def cal_nmi(list1, list2):
    ''' Compute Normalized Mutual Information 
    '''
    nmi_list = []
    for li1, li2 in zip(list1, list2):
        nmi_list.append(normalized_mutual_info_score(li1, li2))
    return np.mean(nmi_list)



def collate_self_train(batch):
    # Separate the input sequences and target values
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    inputs_cluster = [torch.tensor(item[2]) for item in batch]

    inputs = torch.stack(inputs, dim=0)
    inputs_cluster = torch.stack(inputs_cluster, dim=0)
    
    # Convert the targets to a tensor
    targets = torch.tensor(targets)
    
    return inputs, targets, inputs_cluster


def collate_self_test(batch):
    batch_mod = {'sketch_img': [], 'sketch_boxes': [], 'sketch_path': [],
                 'positive_img': [], 'positive_boxes': [], 'positive_path': [],
                 }

    for i_batch in batch:
        batch_mod['sketch_img'].append(i_batch['sketch_img'])
        batch_mod['sketch_path'].append(i_batch['sketch_path'])
        batch_mod['positive_img'].append(i_batch['positive_img'])
        batch_mod['positive_path'].append(i_batch['positive_path'])
        batch_mod['sketch_boxes'].append(torch.tensor(i_batch['sketch_boxes']).float())
        batch_mod['positive_boxes'].append(torch.tensor(i_batch['positive_boxes']).float())

    batch_mod['sketch_img'] = torch.stack(batch_mod['sketch_img'], dim=0)
    batch_mod['positive_img'] = torch.stack(batch_mod['positive_img'], dim=0)

    return batch_mod

