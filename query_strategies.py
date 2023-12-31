import numpy as np
import torch
from math import log
import numpy as np
from finch import FINCH
from utils import open_entropy, lab_conv

def eoal_sampling(args, unlabeledloader, Len_labeled_ind_train, model, model_bc, knownclass, use_gpu, cluster_centers=None, cluster_labels=None, first_rd=True, diversity=True):
    
    model.eval()
    model_bc.eval()
    labelArr, queryIndex, entropy_list, y_pred, unk_entropy_list = [], [], [], [], []
    feat_all = torch.zeros([1, 128]).cuda()
    precision, recall = 0, 0

    for batch_idx, (index, (data, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, features = model(data)
        softprobs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(softprobs, 1)
        queryIndex += index
        y_pred += list(np.array(predicted.cpu().data))
        labelArr += list(np.array(labels.cpu().data))
        feat_all = torch.cat([feat_all, features.data],0)
        out_open = model_bc(features)
        out_open = out_open.view(outputs.size(0), 2, -1)

        ####### closed-set entropy score
        entropy_data = open_entropy(out_open)
        entropy_list.append(entropy_data.cpu().data)
        ####### distance-based entropy score
        if not first_rd:
            dists = torch.cdist(features, cluster_centers)
            similarity_scores_cj = torch.softmax(-dists, dim=1)
            pred_ent = -torch.sum(similarity_scores_cj*torch.log(similarity_scores_cj+1e-20), 1)
            unk_entropy_list.append(pred_ent.cpu().data)

    entropy_list = torch.cat(entropy_list).cpu()
    entropy_list = entropy_list / log(2)

    y_pred = np.array(y_pred)
    labelArr = torch.tensor(labelArr)
    labelArr_k = labelArr[y_pred < args.known_class]
    
    if not first_rd:
        unk_entropy_list = torch.cat(unk_entropy_list).cpu()
        unk_entropy_list = unk_entropy_list / log(len(cluster_centers))
        entropy_list = entropy_list - unk_entropy_list
        
    embeddings = feat_all[1:].cpu().numpy()
    embeddings_k = embeddings[y_pred < args.known_class]

    uncertaintyArr_k = entropy_list[y_pred < args.known_class]
    queryIndex = torch.tensor(queryIndex)
    queryIndex_k = queryIndex[y_pred < args.known_class]
    
    if not diversity:
        sorted_idx = uncertaintyArr_k.sort()[1][:args.query_batch]
        selected_idx = queryIndex_k[sorted_idx]
        selected_gt = labelArr_k[sorted_idx]
        selected_gt = selected_gt.numpy()
        selected_idx = selected_idx.numpy()

    else:        
        labels_c, num_clust, _ = FINCH(embeddings_k, req_clust= len(knownclass), verbose=True)
        tmp_var = 0
        while num_clust[tmp_var] > args.query_batch:
            tmp_var += 1
        cluster_labels = labels_c[:, tmp_var]
        num_clusters = num_clust[tmp_var]

        rem = min(args.query_batch, len(queryIndex_k))
        num_per_cluster = int(rem/num_clusters)
        selected_idx = []
        selected_gt = []

        ax = [0 for i in range(num_clusters)]
        while rem > 0:
            print("Remaining Budget to Sample:  ", rem)
            for cls in range(num_clusters):
                temp_ent = uncertaintyArr_k[cluster_labels == cls]
                temp_index = queryIndex_k[cluster_labels == cls]
                temp_gt = labelArr_k[cluster_labels == cls]
                if rem >= num_per_cluster:
                    sorted_idx = temp_ent.sort()[1][ax[cls]:ax[cls]+min(num_per_cluster, len(temp_ent))]
                    ax[cls] += len(sorted_idx)
                    rem -= len(sorted_idx)
                else:
                    sorted_idx = temp_ent.sort()[1][ax[cls]:ax[cls]+min(rem, len(temp_ent))]
                    ax[cls] += len(sorted_idx)
                    rem -= len(sorted_idx)
                q_idxs = temp_index[sorted_idx.cpu()]
                selected_idx.extend(list(q_idxs.numpy()))
                gt_cls = temp_gt[sorted_idx.cpu()]
                selected_gt.extend(list(gt_cls.numpy()))
        print("clustering finished")
        selected_gt = np.array(selected_gt)
        selected_idx = np.array(selected_idx)
    
    if len(selected_gt) < args.query_batch:
        rem_budget = args.query_batch - len(set(selected_idx))
        print("Not using all the budget...")
        uncertaintyArr_u = entropy_list[y_pred >= args.known_class]
        queryIndex_u = queryIndex[y_pred >= args.known_class]
        queryIndex_u = np.array(queryIndex_u)
        labelArr_u = labelArr[y_pred >= args.known_class]
        labelArr_u = np.array(labelArr_u)
        tmp_data = np.vstack((queryIndex_u, labelArr_u)).T
        print("Choosing from the K+1 classifier's rejected samples...")
        sorted_idx_extra = uncertaintyArr_u.sort()[1][:rem_budget]
        tmp_data = tmp_data.T
        rand_idx = tmp_data[0][sorted_idx_extra.cpu().numpy()]
        rand_LabelArr = tmp_data[1][sorted_idx_extra.cpu().numpy()]
        selected_gt = np.concatenate((selected_gt, rand_LabelArr))
        selected_idx = np.concatenate((selected_idx, rand_idx))
    
    precision = len(np.where(selected_gt < args.known_class)[0]) / len(selected_gt)
    recall = (len(np.where(selected_gt < args.known_class)[0]) + Len_labeled_ind_train) / (
                len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return selected_idx[np.where(selected_gt < args.known_class)[0]], selected_idx[np.where(selected_gt >= args.known_class)[0]], precision, recall

def random_sampling(args, unlabeledloader, Len_labeled_ind_train, model, knownclass):
    model.eval()
    queryIndex = []
    labelArr = []
    precision, recall = 0, 0
    for batch_idx, (index, (_, labels)) in enumerate(unlabeledloader):
        labels = lab_conv(knownclass, labels)
        queryIndex += index
        labelArr += list(np.array(labels.cpu().data))

    tmp_data = np.vstack((queryIndex, labelArr)).T
    np.random.shuffle(tmp_data)
    tmp_data = tmp_data.T
    queryIndex = tmp_data[0][:args.query_batch]
    labelArr = tmp_data[1]
    queryLabelArr = tmp_data[1][:args.query_batch]
    precision = len(np.where(queryLabelArr < args.known_class)[0]) / len(queryLabelArr)
    recall = (len(np.where(queryLabelArr < args.known_class)[0]) + Len_labeled_ind_train) / (len(np.where(labelArr < args.known_class)[0]) + Len_labeled_ind_train)
    return queryIndex[np.where(queryLabelArr < args.known_class)[0]], queryIndex[np.where(queryLabelArr >= args.known_class)[0]], precision, recall
