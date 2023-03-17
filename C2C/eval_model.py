import copy
import torch
import numpy as np
import pandas as pd
import albumentations
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2, ToTensor

from C2C.dataloader import *
from C2C.models.resnet import *

def eval_model(model, dataloaders, dataset_sizes, data_transforms):
    """ Pass all the patches in a WSI for validation prediction
    """
    pred_list = []
    pred_fname = []
    running_corrects = 0
    with torch.no_grad():
        # for im, im_list in tqdm(test_images.items()):
        for i, (inputs, labels, inputs_cluster) in enumerate(tqdm(dataloaders, desc="Testing data", total=len(dataloaders))):   

            # td = WSIDataloader(im_list, transform=data_transforms)
            # tdl = torch.utils.data.DataLoader(td, batch_size=128,
            #                                  shuffle=False, num_workers=0)

            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs, outputs_patch, outputs_attn = model(inputs)
            
            _, preds = torch.max(outputs, 1)
            print(preds)
            # Y_prob = compute_attn_df(tdl, model)
            # _, t_pred = torch.max(Y_prob, 1)
            # label = test_images_label[im]
            running_corrects += torch.sum(preds == labels)
            # pred_list.append(t_pred)
            # pred_fname.append(im)
            # pred_list += [t_pred]*len(im_list)
            # pred_fname += [im]*len(im_list)

    epoch_acc = running_corrects.double() / dataset_sizes
    # pred_df = pd.DataFrame({'wsi': pred_fname, 'prediction': pred_list})
    # pred_df['actual'] = pred_df['wsi'].apply(lambda x: test_images_label[x])

    # print('Test Accuracy: ', sum(pred_df['actual']==pred_df['prediction'])/pred_df.shape[0])  
    print("Test Accuracy: ", epoch_acc.item())          
    
    # Confusion Matrix
    # label_map = {0: 'Normal', 1: 'Disease'}
    # actual_label = pd.Series([label_map[x] for x in pred_df['actual'].tolist()], name='Actual')
    # predicted_label = pd.Series([label_map[x] for x in pred_list], name='Predicted')
    # print(pd.crosstab(actual_label, predicted_label))
    
    # return sum(pred_df['actual']==pred_df['prediction'])/pred_df.shape[0]
    return epoch_acc
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def compute_attn_df(tdl, model, track_running_stats=True):
    """ Pass through all the patches in a WSI for validation prediction
    """
    model = copy.deepcopy(model)
    enc_attn = EncAttn(model)
    enc_attn.eval()
    classifier_layer = model.classifier
    
    # Perform WSI-Batch Normalization for validation, switch off when transfer learning 
    # For batch size=1 - WSI normalization during testing phase works better
    if track_running_stats:
        for m in enc_attn.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.track_running_stats = False
        
    attn_rep = torch.tensor([])
    inputs_rep = torch.tensor([])
    fname_list = []

    for i, sample in enumerate(tdl):
        attn_rep_instance, inp_rep_instance = enc_attn(sample[0].cuda())
        attn_rep_instance = attn_rep_instance.detach()
        inp_rep_instance = inp_rep_instance.detach()
        if len(inputs_rep) == 0:
            inputs_rep = inp_rep_instance
            attn_rep = attn_rep_instance
        else:
            inputs_rep = torch.cat((inputs_rep, inp_rep_instance))
            attn_rep = torch.cat((attn_rep, attn_rep_instance))
        fname_list += list(sample[1])

    A = torch.transpose(attn_rep, 1, 0)
    A = F.softmax(A, dim=1)
    # print("A --> ", A)
    M = torch.mm(A, inputs_rep)
    # print("M --> ", M)
    Y_prob = classifier_layer(M)
    print("Y_prob --> ", Y_prob)   
        
    # return torch.max(Y_prob, 1)[1].item(), attn_rep.cpu().numpy().flatten()
    return Y_prob

def eval_test(model, df, data_transforms):
    """ 
    Parameters:
        model - Trained model
        df - dataframe containing following columns:
            1. path - path of patches
            2. wsi - wsi identifier            
            optional - 
            3. label - positive or negative class - if label column 
            is not provided performance is not reported
                
    returns:
        df - dataframe with prediction for each WSI
    """
    
    test_images = dict(df.groupby('wsi')['path'].apply(list))
    pred_list = []
    pred_fname = []
    pred_attn = []
    pred_path = []
    with torch.no_grad():
        for im, im_list in tqdm(test_images.items()):            
            td = WSIDataloader(im_list, transform=data_transforms)
            tdl = torch.utils.data.DataLoader(td, batch_size=128,
                                             shuffle=False, num_workers=0)
            t_pred, attn_rep = compute_attn_df(tdl, model)
            pred_list += [t_pred]*len(im_list)
            pred_fname += [im]*len(im_list)
            pred_attn += list(attn_rep)
            pred_path += im_list

    pred_df = pd.DataFrame({'wsi': pred_fname, 'prediction': pred_list,\
                            'attention': pred_attn, 'path': pred_path})
    
    if 'label' in df.columns:
        test_images_label = dict(df.groupby('wsi')['label'].apply(max))    
        pred_df['actual'] = pred_df['wsi'].apply(lambda x: test_images_label[x])
        pred_wsi_df = pred_df[['wsi', 'prediction', 'actual']].drop_duplicates()
        print('Test Accuracy: ', sum(pred_wsi_df['actual']==pred_wsi_df['prediction'])/pred_wsi_df.shape[0])            
        
    return pred_df