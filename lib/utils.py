#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal, mnist_noniid_lt
from sampling import femnist_iid, femnist_noniid, femnist_noniid_unequal, femnist_noniid_lt
from sampling import cifar_iid, cifar100_noniid, cifar10_noniid, cifar100_noniid_lt, cifar10_noniid_lt
import femnist
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cosine
from collections import defaultdict

trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std=[0.229, 0.224, 0.225])])
trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                             std=[0.229, 0.224, 0.225])])
trans_cifar100_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                               std=[0.267, 0.256, 0.276])])
trans_cifar100_val = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                              std=[0.267, 0.256, 0.276])])

def get_dataset(args, n_list, k_list):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    data_dir = args.data_dir + args.dataset
    if args.dataset == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(args, train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = mnist_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = mnist_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)
                classes_list_gt = classes_list

    elif args.dataset == 'femnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = femnist.FEMNIST(args, data_dir, train=True, download=True,
                                        transform=apply_transform)
        test_dataset = femnist.FEMNIST(args, data_dir, train=False, download=True,
                                       transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = femnist_iid(train_dataset, args.num_users)
            # print("TBD")
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                # user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
                user_groups = femnist_noniid_unequal(args, train_dataset, args.num_users)
                # print("TBD")
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = femnist_noniid(args, args.num_users, n_list, k_list)
                user_groups_lt = femnist_noniid_lt(args, args.num_users, classes_list)

    elif args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=trans_cifar10_train)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=trans_cifar10_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list, classes_list_gt = cifar10_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar10_noniid_lt(args, test_dataset, args.num_users, n_list, k_list, classes_list)

    elif args.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(data_dir, train=True, download=True, transform=trans_cifar100_train)
        test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=trans_cifar100_val)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups, classes_list = cifar100_noniid(args, train_dataset, args.num_users, n_list, k_list)
                user_groups_lt = cifar100_noniid_lt(test_dataset, args.num_users, classes_list)

    return train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != '....':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_sem(w, n_list):
    """
    Returns the average of the weights.
    """
    k = 2
    model_dict = {}
    for i in range(k):
        model_dict[i] = []

    idx = 0
    for i in n_list:
        if i< np.mean(n_list):
            model_dict[0].append(idx)
        else:
            model_dict[1].append(idx)
        idx += 1

    ww = copy.deepcopy(w)
    for cluster_id in model_dict.keys():
        model_id_list = model_dict[cluster_id]
        w_avg = copy.deepcopy(w[model_id_list[0]])
        for key in w_avg.keys():
            for j in range(1, len(model_id_list)):
                w_avg[key] += w[model_id_list[j]][key]
            w_avg[key] = torch.true_divide(w_avg[key], len(model_id_list))
            # w_avg[key] = torch.div(w_avg[key], len(model_id_list))
        for model_id in model_id_list:
            for key in ww[model_id].keys():
                ww[model_id][key] = w_avg[key]

    return ww

def average_weights_per(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:2] != 'fc':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            # w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def average_weights_het(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w)
    for key in w[0].keys():
        if key[0:4] != 'fc2.':
            for i in range(1, len(w)):
                w_avg[0][key] += w[i][key]
            # w_avg[0][key] = torch.true_divide(w_avg[0][key], len(w))
            w_avg[0][key] = torch.div(w_avg[0][key], len(w))
            for i in range(1, len(w)):
                w_avg[i][key] = w_avg[0][key]
    return w_avg

def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def proto_aggregation(local_protos_list):
    agg_protos_label = dict()
    for idx in local_protos_list:
        local_protos = local_protos_list[idx]
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for [label, proto_list] in agg_protos_label.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            agg_protos_label[label] = [proto / len(proto_list)]
        else:
            agg_protos_label[label] = [proto_list[0].data]

    return agg_protos_label


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.rounds}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.train_ep}\n')
    return

def calculate_vag(client_protos, global_protos_base, val_loader, model, device, class_list):
    """
    Calculate Validation Accuracy Gain (VAG) for a client
    
    Args:
        client_protos: Client's submitted prototypes
        global_protos_base: Baseline global prototypes (without this client)
        val_loader: Validation data loader
        model: Model architecture for classification
        device: Computation device
        class_list: List of classes
    
    Returns:
        VAG score (float)
    """
    # Create temporary global prototypes with client's contribution
    global_protos_with_client = {}
    for label in class_list:
        if label in client_protos:
            global_protos_with_client[label] = client_protos[label]
        elif label in global_protos_base:
            global_protos_with_client[label] = global_protos_base[label]
    
    # Calculate accuracy with baseline
    acc_base = evaluate_prototypes(global_protos_base, val_loader, device, class_list)
    
    # Calculate accuracy with client's contribution
    acc_with_client = evaluate_prototypes(global_protos_with_client, val_loader, device, class_list)
    
    return acc_with_client - acc_base


def evaluate_prototypes(prototypes, data_loader, device, class_list):
    """Evaluate prototype-based classifier accuracy"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            
            # For each sample, find nearest prototype
            for i in range(len(data)):
                sample = data[i]
                min_dist = float('inf')
                pred_label = -1
                
                for label in class_list:
                    if label in prototypes:
                        dist = torch.norm(sample - prototypes[label])
                        if dist < min_dist:
                            min_dist = dist
                            pred_label = label
                
                all_preds.append(pred_label)
                all_labels.append(target[i].item())
    
    return accuracy_score(all_labels, all_preds)


def calculate_pds(client_protos, all_client_protos, class_frequencies):
    """
    Calculate Prototype Diversity Score (PDS)
    
    Args:
        client_protos: Current client's prototypes
        all_client_protos: Dict mapping client_id -> prototypes
        class_frequencies: Dict mapping class -> frequency count
    
    Returns:
        PDS score (float)
    """
    pds_score = 0.0
    
    for label, proto in client_protos.items():
        # Component 1: Class rarity score
        rarity_score = 1.0 / max(class_frequencies.get(label, 1), 1)
        
        # Component 2: Embedding space sparsity
        distances = []
        for other_client_id, other_protos in all_client_protos.items():
            if label in other_protos:
                dist = torch.norm(proto - other_protos[label]).item()
                distances.append(dist)
        
        sparsity_score = np.mean(distances) if distances else 0.0
        
        pds_score += rarity_score + sparsity_score
    
    return pds_score


def calculate_cosine_anomaly(client_update_vector, global_update_vector):
    """
    Calculate Cosine Anomaly Score (CAS)
    
    Args:
        client_update_vector: Client's prototype update direction
        global_update_vector: Aggregated global update direction
    
    Returns:
        Anomaly score (0 = aligned, 1 = opposite)
    """
    if torch.norm(client_update_vector) == 0 or torch.norm(global_update_vector) == 0:
        return 1.0  # Maximum anomaly if either vector is zero
    
    cos_sim = torch.dot(client_update_vector.flatten(), global_update_vector.flatten()) / \
              (torch.norm(client_update_vector) * torch.norm(global_update_vector))
    
    return 1.0 - cos_sim.item()


def detect_outliers_clustering(all_protos, class_label, eps=0.5, min_samples=2):
    """
    Detect outlier prototypes using DBSCAN clustering
    
    Args:
        all_protos: List of (client_id, prototype) tuples for a specific class
        class_label: The class being analyzed
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN min_samples parameter
    
    Returns:
        List of client_ids identified as outliers
    """
    if len(all_protos) < min_samples:
        return []
    
    proto_vectors = np.array([p[1].cpu().numpy().flatten() for p in all_protos])
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(proto_vectors)
    
    outliers = []
    for idx, (client_id, _) in enumerate(all_protos):
        if clustering.labels_[idx] == -1:  # -1 indicates outlier in DBSCAN
            outliers.append(client_id)
    
    return outliers

def agg_func_dcaa(protos, args, client_weights=None, client_ids=None):
    """
    Dynamic Contribution-Aware Aggregation for FedProto
    Returns: dict mapping label -> aggregated prototype tensor
    """
    if client_weights is None:
        client_weights = {cid: 1.0 for cid, _ in protos}
    
    if client_ids is not None:
        protos = [(cid, cp) for cid, cp in protos if cid in client_ids]
    
    agg_protos_label = {}
    
    for idx, (client_id, client_protos) in enumerate(protos):
        for label in client_protos.keys():
            if label not in agg_protos_label:
                agg_protos_label[label] = []
            
            weight = client_weights.get(client_id, 1.0)
            agg_protos_label[label].append((weight, client_protos[label]))
    
    # Weighted aggregation
    for label in agg_protos_label.keys():
        if len(agg_protos_label[label]) > 0:
            total_weight = sum([w for w, _ in agg_protos_label[label]])
            
            if total_weight > 0:
                proto = torch.zeros_like(agg_protos_label[label][0][1])
                for weight, proto_tensor in agg_protos_label[label]:
                    proto += (weight / total_weight) * proto_tensor
                
                agg_protos_label[label] = proto
            else:
                # If all weights are zero, use simple average
                proto = torch.zeros_like(agg_protos_label[label][0][1])
                for _, proto_tensor in agg_protos_label[label]:
                    proto += proto_tensor / len(agg_protos_label[label])
                
                agg_protos_label[label] = proto
    
    return agg_protos_label

def proto_aggregation_secondary(local_model_list, test_dataset, user_groups_lt, 
                                global_protos, args, idxs_users):
    """
    Evaluate models with and without global prototypes
    
    Returns:
        acc_test: Test accuracy with global prototypes
        acc_test_local: Test accuracy without global prototypes (local only)
    """
    # Implementation similar to test_inference_new_het_lt
    acc_list_g = []
    acc_list_l = []
    
    for idx in idxs_users:
        local_model = local_model_list[idx]
        local_model.eval()
        
        # Test with global prototypes
        acc_g = test_inference_with_protos(args, local_model, test_dataset, 
                                           user_groups_lt[idx], global_protos)
        acc_list_g.append(acc_g)
        
        # Test without global prototypes (local only)
        acc_l = test_inference_local(args, local_model, test_dataset, 
                                     user_groups_lt[idx])
        acc_list_l.append(acc_l)
    
    return np.mean(acc_list_g), np.mean(acc_list_l)


def test_inference_with_protos(args, model, dataset, idxs, global_protos):
    """Test inference using model + global prototypes"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = args.device
    testloader = DataLoader(DatasetSplit(dataset, idxs), 
                           batch_size=128, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get model outputs
            outputs, protos = model(images)
            
            # Enhance with global prototypes if available
            if global_protos:
                # Prototype-based prediction enhancement
                proto_scores = []
                for label in range(args.num_classes):
                    if label in global_protos:
                        proto = global_protos[label]
                        # Calculate distance to prototype
                        dist = torch.norm(protos - proto.unsqueeze(0), dim=1)
                        proto_scores.append(-dist)
                    else:
                        proto_scores.append(torch.zeros(len(images)).to(device))
                
                proto_scores = torch.stack(proto_scores, dim=1)
                outputs = outputs + 0.5 * proto_scores  # Combine model + proto scores
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct / total
    return accuracy


def test_inference_local(args, model, dataset, idxs):
    """Test inference using only local model (no prototypes)"""
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0
    
    device = args.device
    testloader = DataLoader(DatasetSplit(dataset, idxs), 
                           batch_size=128, shuffle=False)
    
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            
            # Get model outputs only
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]  # Get logits if model returns tuple
            
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
    
    accuracy = correct / total
    return accuracy

class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)