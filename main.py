import os
import argparse
import datetime
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import ResNet18, ResClassifier_MME
import query_strategies
import datasets
from utils import AverageMeter, get_splits, open_entropy, entropic_bc_loss, reg_loss, lab_conv, unknown_clustering

parser = argparse.ArgumentParser("Entropic Open-set Active Learning")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar100', choices=['cifar100', 'cifar10', 'tinyimagenet'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 0)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=300)
parser.add_argument('--max-query', type=int, default=11)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--stepsize', type=int, default=60)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
parser.add_argument('--query-strategy', type=str, default='eoal', choices=['random', 'eoal'])

# model
parser.add_argument('--model', type=str, default='resnet18')
# misc
parser.add_argument('--eval-freq', type=int, default=300)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--known-T', type=float, default=0.5)
parser.add_argument('--unknown-T', type=float, default=0.5)
parser.add_argument('--modelB-T', type=float, default=1)
parser.add_argument('--init-percent', type=int, default=16)
parser.add_argument('--diversity', type=int, default=1)
parser.add_argument('--pareta-alpha', type=float, default=0.8)
parser.add_argument('--reg-w', type=float, default=0.1)
parser.add_argument('--w-unk-cls', type=int, default=1)
parser.add_argument('--w-ent', type=float, default=1)

args = parser.parse_args()

def main():
    seed = 1
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    knownclass = get_splits(args.dataset, seed, args.known_class)  
    print("Known Classes", knownclass)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    testloader, unlabeledloader = dataset.testloader, dataset.unlabeledloader
    trainloader_A, trainloader_B = dataset.trainloader, dataset.trainloader
    trainloader_C = None   # init negativeloader none
    invalidList = []
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

    print("Creating model: {}".format(args.model))
    start_time = time.time()

    Acc = {}
    Err = {}
    Precision = {}
    Recall = {}
    for query in tqdm(range(args.max_query)):
        # Model initialization
        model_A = ResNet18(n_class=dataset.num_classes+1)
        model_B = ResNet18(n_class=dataset.num_classes)

        if use_gpu:
            model_A = nn.DataParallel(model_A).cuda()
            model_B = nn.DataParallel(model_B).cuda()
            model_bc = ResClassifier_MME(num_classes=2 * (dataset.num_classes),
                        norm=False, input_size=128).cuda()
            
        criterion_xent = nn.CrossEntropyLoss()
        optimizer_model_A = torch.optim.SGD(model_A.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        optimizer_model_B = torch.optim.SGD(model_B.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        params_bc = list(model_bc.parameters())
        optim_C = torch.optim.SGD(params_bc, lr=args.lr_model, momentum=0.9, weight_decay=0.0005, nesterov=True)

        if args.stepsize > 0:
            scheduler_A = lr_scheduler.StepLR(optimizer_model_A, step_size=args.stepsize, gamma=args.gamma)
            scheduler_B = lr_scheduler.StepLR(optimizer_model_B, step_size=args.stepsize, gamma=args.gamma)

        # Model training 
        for epoch in tqdm(range(args.max_epoch)):
            # Train model A for detecting unknown classes
            if query > 0:
                cluster_centers, _, cluster_labels, cluster_indices = unknown_clustering(args, model_A, model_bc, trainloader_C, use_gpu, knownclass)
                train_A(model_A, model_bc, criterion_xent,
                optimizer_model_A, optim_C,
                trainloader_A, invalidList, use_gpu, dataset.num_classes, epoch, knownclass, cluster_centers, cluster_labels, cluster_indices)
            else:
                train_A(model_A, model_bc, criterion_xent,
                    optimizer_model_A, optim_C,
                    trainloader_A, invalidList, use_gpu, dataset.num_classes, epoch, knownclass)
            # Train model B for classifying known classes
            train_B(model_B, criterion_xent,
                optimizer_model_B,
                trainloader_B, use_gpu, knownclass)

            if args.stepsize > 0:
                scheduler_A.step()
                scheduler_B.step()

            if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
                print("==> Test")
                acc_A, err_A = test(model_A, testloader, use_gpu, knownclass)
                acc_B, err_B = test(model_B, testloader, use_gpu, knownclass)
                print("Model_A | Accuracy (%): {}\t Error rate (%): {}".format(acc_A, err_A))
                print("Model_B | Accuracy (%): {}\t Error rate (%): {}".format(acc_B, err_B))

        # Record results
        acc, err = test(model_B, testloader, use_gpu, knownclass)
        Acc[query], Err[query] = float(acc), float(err)
        
        queryIndex = []
        if args.query_strategy == "random":
            queryIndex, invalidIndex, Precision[query], Recall[query] = query_strategies.random_sampling(args, unlabeledloader, len(labeled_ind_train), model_B)
        elif args.query_strategy == "eoal":
            if query == 0:
                queryIndex, invalidIndex, Precision[query], Recall[query] = query_strategies.eoal_sampling(args, unlabeledloader, len(labeled_ind_train), model_A, model_bc, knownclass, use_gpu, None,None, True, args.diversity)            
            else:
                queryIndex, invalidIndex, Precision[query], Recall[query] = query_strategies.eoal_sampling(args, unlabeledloader, len(labeled_ind_train), model_A, model_bc, knownclass, use_gpu, cluster_centers, cluster_labels, False, args.diversity)    
        
        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train)-set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        invalidList = list(invalidList) + list(invalidIndex)

        print("Query Strategy: "+args.query_strategy+" | Query Budget: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train)))
        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=list(set(unlabeled_ind_train) - set(invalidList)), labeled_ind_train=labeled_ind_train+invalidList,
        )
        trainloader_A, testloader, unlabeledloader = dataset.trainloader, dataset.testloader, dataset.unlabeledloader
        B_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        )
        trainloader_B = B_dataset.trainloader
        C_dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, knownclass=knownclass, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=invalidList,
        )
        trainloader_C = C_dataset.trainloader

    all_accuracies.append(Acc)
    all_precisions.append(Precision)
    all_recalls.append(Recall)
    print("Accuracies", all_accuracies)
    print("Precisions", all_precisions)
    print("Recalls", all_recalls)
   
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

def train_A(model, model_bc, criterion_xent,
        optimizer_model, optimizer_bc,
        trainloader, invalidList, use_gpu, num_classes, epoch, knownclass, cluster_centers=None, cluster_labels=None, cluster_indices=None):
    model.train()
    model_bc.train()
    xent_losses = AverageMeter()
    open_losses = AverageMeter()
    k_losses = AverageMeter()
    losses = AverageMeter()

    known_T = args.known_T
    unknown_T = args.unknown_T
    ent_list = []
    labelArr = []
    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        # Reduce temperature
        T = torch.tensor([known_T]*labels.shape[0], dtype=float)
        labels = lab_conv(knownclass, labels)
        labelArr += list(np.array(labels.cpu().data))
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, features = model(data)
        out_open = model_bc(features)
        out_open = out_open.view(outputs.size(0), 2, -1) 
        ent = open_entropy(out_open)
        
        ent_list.append(ent.cpu().data)
        labels_unk = []
        for i in range(len(labels)):
            # Annotate "unknown"
            if index[i] in invalidList:
                T[i] = unknown_T
                tmp_idx = index[i]
                loc = torch.where(cluster_indices == tmp_idx)[0]
                labels_unk += list(np.array(cluster_labels[loc].cpu().data))
        labels_unk = torch.tensor(labels_unk).cuda()
        T = T.cuda()
        open_loss_pos, open_loss_neg, open_loss_pos_ood, open_loss_neg_ood = entropic_bc_loss(out_open, labels, args.pareta_alpha, num_classes, len(invalidList), args.w_ent)
        
        if len(invalidList) > 0:
            regu_loss, _, _ = reg_loss(features, labels, cluster_centers, labels_unk, num_classes) 
            loss_open = 0.5 * (open_loss_pos + open_loss_neg) + 0.5 * (open_loss_pos_ood + open_loss_neg_ood)
        else:
            loss_open = 0.5 * (open_loss_pos + open_loss_neg)

        outputs = outputs / T.unsqueeze(1)
        loss_xent = criterion_xent(outputs, labels)
        if len(invalidList) > 0:
            loss = loss_xent + loss_open + args.reg_w * regu_loss
        else:
            loss = loss_xent + loss_open

        optimizer_model.zero_grad()
        optimizer_bc.zero_grad()
        loss.backward()
        optimizer_model.step()
        optimizer_bc.step()
        
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        open_losses.update(loss_open.item(), labels.size(0))
        if len(invalidList) > 0:
            k_losses.update(regu_loss.item(), labels.size(0))
        
    labelArr = torch.tensor(labelArr)
    ent_list = torch.cat(ent_list).cpu()
   
    if epoch%50 == 0:
        print(f" loss: {losses.avg} xent_loss: {xent_losses.avg} open_loss: {open_losses.avg} regu_loss: {k_losses.avg}")

def train_B(model, criterion_xent, optimizer_model, trainloader, use_gpu, knownclass):
    model.train()
    xent_losses = AverageMeter()
    losses = AverageMeter()

    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        labels = lab_conv(knownclass, labels)
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        outputs, _ = model(data)
        loss_xent = criterion_xent(outputs, labels)
        loss = loss_xent 
        optimizer_model.zero_grad()
        loss.backward()
        optimizer_model.step()
    
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))

def test(model, testloader, use_gpu, knownclass):
    model.eval()
    correct, total = 0, 0
    
    with torch.no_grad():
        for index, (data, labels) in testloader:
            labels = lab_conv(knownclass, labels)
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            outputs, _ = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

if __name__ == '__main__':
    main()





