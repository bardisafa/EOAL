import torch
import torchvision
from torch.utils.data import SubsetRandomSampler
import random
import transforms
from torch.utils.data import Dataset
from PIL import Image
from utils import lab_conv

known_class = -1
init_percent = -1

class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.X)

class CIFAR100(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.CIFAR100("./data/cifar100", train=True, download=True, transform=transform_train)
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train),
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train),
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        testset = torchvision.datasets.CIFAR100("./data/cifar100", train=False, download=True, transform=transform_test)
        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test),
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            c = lab_conv(knownclass_list, [c])
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            c = lab_conv(knownclass_list, [c])
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind

class CIFAR10(object):
    def __init__(self, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None,
                 labeled_ind_train=None):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.CIFAR10("./data/cifar10", train=True, download=True, transform=transform_train)
        if unlabeled_ind_train == None and labeled_ind_train == None:
            if is_mini:
                labeled_ind_train, unlabeled_ind_train = self.filter_known_unknown_10percent(trainset)
                self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train
            else:
                labeled_ind_train = self.filter_known_unknown(trainset)
                self.labeled_ind_train = labeled_ind_train
        else:
            self.labeled_ind_train, self.unlabeled_ind_train = labeled_ind_train, unlabeled_ind_train

        if is_filter:
            print("openset here!")
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(labeled_ind_train)
            )
            unlabeledloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(unlabeled_ind_train)
            )
        else:
            trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True,
                num_workers=num_workers, pin_memory=pin_memory
            )

        testset = torchvision.datasets.CIFAR10("./data/cifar10", train=False, download=True, transform=transform_test)
        filter_ind_test = self.filter_known_unknown(testset)
        self.filter_ind_test = filter_ind_test

        if is_filter:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory,
                sampler=SubsetRandomSampler(filter_ind_test)
            )
        else:
            testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=pin_memory
            )

        self.trainloader = trainloader
        if is_filter: self.unlabeledloader = unlabeledloader
        self.testloader = testloader
        self.num_classes = known_class

    def filter_known_unknown(self, dataset):
        filter_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            c = lab_conv(knownclass_list, [c])
            if c < known_class:
                filter_ind.append(i)
        return filter_ind

    def filter_known_unknown_10percent(self, dataset):
        filter_ind = []
        unlabeled_ind = []
        for i in range(len(dataset.targets)):
            c = dataset.targets[i]
            c = lab_conv(knownclass_list, [c])
            if c < known_class:
                filter_ind.append(i)
            else:
                unlabeled_ind.append(i)

        random.shuffle(filter_ind)
        labeled_ind = filter_ind[:len(filter_ind) * init_percent // 100]
        unlabeled_ind = unlabeled_ind + filter_ind[len(filter_ind) * init_percent // 100:]
        return labeled_ind, unlabeled_ind

__factory = {
    'cifar100': CIFAR100,
    'cifar10': CIFAR10}

def create(name, known_class_, knownclass, init_percent_, batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train=None, labeled_ind_train=None):
    global known_class, init_percent, knownclass_list
    known_class = known_class_
    init_percent = init_percent_
    knownclass_list = knownclass
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](batch_size, use_gpu, num_workers, is_filter, is_mini, unlabeled_ind_train, labeled_ind_train)