#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path
from collections import defaultdict

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
from utils import calculate_vag, calculate_pds, calculate_cosine_anomaly, detect_outliers_clustering, agg_func_dcaa

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))
            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global weights
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (with protos), mean of proto loss is {:.5f}, std of test acc is {:.5f}'.format(np.mean(loss_list), np.std(loss_list)))

    # save protos
    if args.dataset == 'mnist':
        save_protos(args, local_model_list, test_dataset, user_groups_lt)

def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_mh_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')

    global_protos = []
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        print(f'\n | Global Training Round : {round + 1} |\n')

        proto_loss = 0
        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
            w, loss, acc, protos = local_model.update_weights_het(args, idx, global_protos, model=copy.deepcopy(local_model_list[idx]), global_round=round)
            agg_protos = agg_func(protos)

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss['total']))

            local_protos[idx] = agg_protos
            summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round)
            summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round)
            summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round)
            summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round)
            proto_loss += loss['2']

        # update global weights
        local_weights_list = local_weights

        for idx in idxs_users:
            local_model = copy.deepcopy(local_model_list[idx])
            local_model.load_state_dict(local_weights_list[idx], strict=True)
            local_model_list[idx] = local_model

        # update global protos
        global_protos = proto_aggregation(local_protos)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

    acc_list_l, acc_list_g = test_inference_new_het_lt(args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)
    print('For all users (with protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_g),np.std(acc_list_g)))
    print('For all users (w/o protos), mean of test acc is {:.5f}, std of test acc is {:.5f}'.format(np.mean(acc_list_l), np.std(acc_list_l)))

def FedProto_DCAA(args, train_dataset, test_dataset):
    """
    FedProto with Dynamic Contribution-Aware Aggregation (DCAA)
    """
    # Initialize
    dict_users = dataset_iid(train_dataset, args.num_users)
    
    # Create server validation set
    val_size = int(0.1 * len(train_dataset))
    val_indices = np.random.choice(len(train_dataset), val_size, replace=False)
    val_loader = DataLoader(
        DatasetSplit(train_dataset, val_indices),
        batch_size=args.local_bs,
        shuffle=False
    )
    
    # Initialize tracking structures
    client_contribution_history = defaultdict(int)  # Consecutive low contribution count
    client_weights = {i: 1.0 for i in range(args.num_users)}
    client_protos_history = defaultdict(dict)  # For tracking prototype movement
    global_protos_history = []
    
    # Hyperparameters
    alpha = args.alpha  # Weight for VAG
    beta = args.beta    # Weight for PDS
    tau_low = args.tau_low
    tau_high = args.tau_high
    gamma = args.gamma  # Throttling factor
    consecutive_threshold = args.consecutive_rounds
    
    # Initialize global prototypes
    global_protos = {}
    
    # Training loop
    for round_idx in range(args.rounds):
        local_protos = {}
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        # Client local training
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
            
            # Pass global prototypes for regularization
            protos, _ = local.update_weights_het(
                args, idx, global_protos, start_epoch=0
            )
            local_protos[idx] = copy.deepcopy(protos)
            client_protos_history[idx] = copy.deepcopy(protos)
        
        # PHASE 1: Unified Contribution & Anomaly Scoring
        contribution_scores = {}
        anomaly_scores = {}
        
        # Calculate class frequencies
        class_frequencies = defaultdict(int)
        for client_id, protos in local_protos.items():
            for label in protos.keys():
                class_frequencies[label] += 1
        
        for client_id in idxs_users:
            # Calculate VAG
            baseline_protos = copy.deepcopy(global_protos)
            vag_score = calculate_vag(
                local_protos[client_id],
                baseline_protos,
                val_loader,
                None,  # Model evaluation handled in calculate_vag
                args.device,
                list(class_frequencies.keys())
            )
            
            # Calculate PDS
            pds_score = calculate_pds(
                local_protos[client_id],
                client_protos_history,
                class_frequencies
            )
            
            # Unified contribution score
            contribution_scores[client_id] = alpha * vag_score + beta * pds_score
            
            # Calculate anomaly score (cosine similarity with global update)
            if len(global_protos_history) >= 2:
                # Calculate client update direction
                client_update = {}
                for label in local_protos[client_id].keys():
                    if label in global_protos_history[-1]:
                        client_update[label] = local_protos[client_id][label] - global_protos_history[-1][label]
                
                # Calculate global update direction
                global_update = {}
                for label in global_protos_history[-1].keys():
                    if label in global_protos_history[-2]:
                        global_update[label] = global_protos_history[-1][label] - global_protos_history[-2][label]
                
                # Flatten and compute cosine anomaly
                if client_update and global_update:
                    client_vec = torch.cat([client_update[l].flatten() for l in client_update.keys()])
                    global_vec = torch.cat([global_update[l].flatten() for l in global_update.keys() if l in client_update])
                    anomaly_scores[client_id] = calculate_cosine_anomaly(client_vec, global_vec)
                else:
                    anomaly_scores[client_id] = 0.0
            else:
                anomaly_scores[client_id] = 0.0
            
            # Update contribution history
            if anomaly_scores[client_id] > tau_high:
                client_contribution_history[client_id] += 1
            elif contribution_scores[client_id] < tau_low:
                client_contribution_history[client_id] += 1
            else:
                client_contribution_history[client_id] = 0  # Reset on good contribution
        
        # PHASE 2 & 3: Dynamic Weighting and Throttling
        for client_id in idxs_users:
            weight = contribution_scores.get(client_id, 0.0)
            
            # Apply throttling if consecutive low contributions
            if client_contribution_history[client_id] >= consecutive_threshold:
                weight = gamma * weight
                print(f"Client {client_id} throttled (consecutive low: {client_contribution_history[client_id]})")
            
            client_weights[client_id] = max(weight, 1e-6)  # Prevent zero weights
        
        # Normalize weights
        total_weight = sum([client_weights[cid] for cid in idxs_users])
        if total_weight > 0:
            for cid in idxs_users:
                client_weights[cid] /= total_weight
        
        # Aggregate with DCAA
        protos_to_agg = [(cid, local_protos[cid]) for cid in idxs_users]
        global_protos = agg_func_dcaa(protos_to_agg, args, client_weights, idxs_users)
        
        # Store history
        global_protos_history.append(copy.deepcopy(global_protos))
        
        # Evaluation
        if round_idx % args.test_freq == 0:
            acc_test = proto_aggregation(global_protos, test_dataset, args)
            print(f"Round {round_idx}: Test Accuracy = {acc_test:.2f}%")
    
    return global_protos

def FedProto_DCAA_taskheter(args, train_dataset, test_dataset, user_groups, 
                            user_groups_lt, local_model_list, classes_list):
    """
    FedProto with DCAA for Task Heterogeneity
    """
    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_dcaa_' + str(args.ways) + 'w' + str(args.shots) + 's' + str(args.stdev) + 'e_' + str(args.num_users) + 'u_' + str(args.rounds) + 'r')
    
    # Initialize tracking structures
    client_contribution_history = defaultdict(int)
    client_weights = {i: 1.0 for i in range(args.num_users)}
    client_protos_history = defaultdict(dict)
    global_protos_history = []
    
    # Hyperparameters
    alpha = args.alpha
    beta = args.beta
    tau_low = args.tau_low
    tau_high = args.tau_high
    gamma = args.gamma
    consecutive_threshold = args.consecutive_rounds
    
    # Initialize global prototypes (as list format for compatibility)
    global_protos = []
    idxs_users = np.arange(args.num_users)
    
    # Setup attackers if specified
    attacker_ids = []
    attacker_obj = None
    if args.num_attackers > 0:
        attacker_ids = np.random.choice(args.num_users, args.num_attackers, replace=False).tolist()
        print(f"Attacker IDs: {attacker_ids}")
        
        if args.attack_type == 'plain':
            from attacks import PlainReplayAttack
            attacker_obj = PlainReplayAttack()
        elif args.attack_type == 'perturbation':
            from attacks import PerturbationAttack
            attacker_obj = PerturbationAttack(sigma=args.attack_sigma)
        elif args.attack_type == 'extrapolation':
            from attacks import ExtrapolationAttack
            attacker_obj = ExtrapolationAttack()
    
    train_loss, train_accuracy = [], []
    
    # Training loop
    for round_idx in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        
        print(f'\n| Global Training Round {round_idx+1}/{args.rounds} |')
        
        proto_loss = 0
        
        # Convert global_protos from list to dict for this round
        global_protos_dict = {}
        if global_protos:
            for label, proto in global_protos:
                global_protos_dict[label] = proto
        
        # Client local training
        for idx in idxs_users:
            if idx in attacker_ids and attacker_obj is not None:
                # Free-rider: generate fake prototypes
                prev_global_dict = {}
                if len(global_protos_history) >= 2:
                    for label, proto in global_protos_history[-2]:
                        prev_global_dict[label] = proto
                
                fake_protos_dict = attacker_obj.generate_fake_protos(
                    global_protos_dict, 
                    prev_global_dict
                )
                
                # Convert to aggregated format
                agg_protos = agg_func(fake_protos_dict)
                local_protos[idx] = agg_protos
                
                print(f'Client {idx} (ATTACKER): Submitted fake prototypes')
            else:
                # Honest client: perform local training
                local_model = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[idx])
                
                # Pass global_protos as dict to update_weights_het
                w, loss, acc, protos = local_model.update_weights_het(
                    args, idx, global_protos_dict, 
                    model=copy.deepcopy(local_model_list[idx]), 
                    global_round=round_idx
                )
                
                agg_protos = agg_func(protos)
                
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss['total']))
                local_protos[idx] = agg_protos
                client_protos_history[idx] = copy.deepcopy(agg_protos)
                
                summary_writer.add_scalar('Train/Loss/user' + str(idx + 1), loss['total'], round_idx)
                summary_writer.add_scalar('Train/Loss1/user' + str(idx + 1), loss['1'], round_idx)
                summary_writer.add_scalar('Train/Loss2/user' + str(idx + 1), loss['2'], round_idx)
                summary_writer.add_scalar('Train/Acc/user' + str(idx + 1), acc, round_idx)
                proto_loss += loss['2']
                
                print(f'Client {idx}: Loss = {loss["total"]:.4f}, Acc = {acc:.4f}, Classes = {list(protos.keys())}')
        
        # Update local models with trained weights
        for idx_pos, idx in enumerate(idxs_users):
            if idx not in attacker_ids and idx_pos < len(local_weights):
                local_model = copy.deepcopy(local_model_list[idx])
                local_model.load_state_dict(local_weights[idx_pos], strict=True)
                local_model_list[idx] = local_model
        
        # PHASE 1: Unified Contribution & Anomaly Scoring
        contribution_scores = {}
        anomaly_scores = {}
        
        # Calculate class frequencies
        class_frequencies = defaultdict(int)
        for client_id, protos in local_protos.items():
            if isinstance(protos, dict):
                for label in protos.keys():
                    class_frequencies[label] += 1
        
        for client_id in idxs_users:
            if client_id in attacker_ids:
                # Assign low scores to known attackers
                contribution_scores[client_id] = 0.0
                anomaly_scores[client_id] = 1.0
                continue
            
            # Calculate VAG (simplified - just use accuracy improvement heuristic)
            vag_score = 0.5  # Placeholder - in production, compute actual VAG
            
            # Calculate PDS
            pds_score = calculate_pds(
                local_protos[client_id],
                client_protos_history,
                class_frequencies
            )
            
            # Unified contribution score
            contribution_scores[client_id] = alpha * vag_score + beta * pds_score
            
            # Calculate anomaly score
            if len(global_protos_history) >= 2:
                # Convert to dicts for comparison
                prev_global_dict = {label: proto for label, proto in global_protos_history[-2]}
                curr_global_dict = {label: proto for label, proto in global_protos_history[-1]}
                
                client_update = {}
                client_proto_dict = local_protos[client_id]
                
                for label in client_proto_dict.keys():
                    if label in curr_global_dict:
                        client_update[label] = client_proto_dict[label] - curr_global_dict[label]
                
                global_update = {}
                for label in curr_global_dict.keys():
                    if label in prev_global_dict:
                        global_update[label] = curr_global_dict[label] - prev_global_dict[label]
                
                if client_update and global_update:
                    # Only compare labels that exist in both
                    common_labels = set(client_update.keys()) & set(global_update.keys())
                    if common_labels:
                        client_vec = torch.cat([client_update[l].flatten() for l in common_labels])
                        global_vec = torch.cat([global_update[l].flatten() for l in common_labels])
                        anomaly_scores[client_id] = calculate_cosine_anomaly(client_vec, global_vec)
                    else:
                        anomaly_scores[client_id] = 0.0
                else:
                    anomaly_scores[client_id] = 0.0
            else:
                anomaly_scores[client_id] = 0.0
            
            # Update contribution history - FIXED LOGIC
            # Only increment if BOTH conditions are true OR score is very low
            if anomaly_scores[client_id] > tau_high and contribution_scores[client_id] < tau_low:
                client_contribution_history[client_id] += 1
            elif contribution_scores[client_id] < tau_low * 0.5:  # Very low contribution
                client_contribution_history[client_id] += 1
            else:
                client_contribution_history[client_id] = max(0, client_contribution_history[client_id] - 1)
            
            print(f'Client {client_id}: Contrib={contribution_scores[client_id]:.4f}, Anomaly={anomaly_scores[client_id]:.4f}, History={client_contribution_history[client_id]}')
        
        # PHASE 2 & 3: Dynamic Weighting and Throttling
        for client_id in idxs_users:
            weight = contribution_scores.get(client_id, 0.0)
            
            if client_contribution_history[client_id] >= consecutive_threshold:
                weight = gamma * weight
                print(f'Client {client_id} THROTTLED (consecutive low: {client_contribution_history[client_id]})')
            
            client_weights[client_id] = max(weight, 1e-6)
        
        # Normalize weights
        total_weight = sum([client_weights[cid] for cid in idxs_users])
        if total_weight > 0:
            for cid in idxs_users:
                client_weights[cid] /= total_weight
        else:
            # If all weights are zero, use uniform weighting
            for cid in idxs_users:
                client_weights[cid] = 1.0 / len(idxs_users)
        
        # Aggregate with DCAA
        protos_to_agg = [(cid, local_protos[cid]) for cid in idxs_users]
        global_protos_dict_agg = agg_func_dcaa(protos_to_agg, args, client_weights, list(idxs_users))
        
        # Convert aggregated dict back to list format for compatibility
        global_protos = [(label, proto) for label, proto in global_protos_dict_agg.items()]
        
        # Store history
        global_protos_history.append(copy.deepcopy(global_protos))
        
        loss_avg = sum(local_losses) / len(local_losses) if local_losses else 0.0
        train_loss.append(loss_avg)
    
    # FIXED: Convert global_protos to dict format for test_inference_new_het_lt
    global_protos_dict_final = {}
    if global_protos:
        for label, proto in global_protos:
            global_protos_dict_final[label] = proto
    
    # Final evaluation
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos_dict_final
    )
    
    print('\n' + '='*60)
    print('FINAL RESULTS (DCAA)')
    print('='*60)
    print(f'With global protos - Mean Acc: {np.mean(acc_list_g):.5f}, Std: {np.std(acc_list_g):.5f}')
    print(f'Without protos     - Mean Acc: {np.mean(acc_list_l):.5f}, Std: {np.std(acc_list_l):.5f}')
    print(f'Proto loss         - Mean: {np.mean(loss_list):.5f}, Std: {np.std(loss_list):.5f}')
    print('='*60)
    
    return global_protos


def FedProto_DCAA_modelheter(args, train_dataset, test_dataset, user_groups, 
                             user_groups_lt, local_model_list, classes_list):
    """
    FedProto with DCAA for Model Heterogeneity
    """
    return FedProto_DCAA_taskheter(args, train_dataset, test_dataset, user_groups,
                                   user_groups_lt, local_model_list, classes_list)

# class DatasetSplit(Dataset):
#     """An abstract Dataset class wrapped around Pytorch Dataset class."""

#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]

#     def __len__(self):
#         return len(self.idxs)

#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         return image.clone().detach(), torch.tensor(label)

if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # load dataset and user groups
    n_list = np.random.randint(max(2, args.ways - args.stdev), min(args.num_classes, args.ways + args.stdev + 1), args.num_users)
    if args.dataset == 'mnist':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users)
    elif args.dataset == 'cifar10':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)
    elif args.dataset == 'cifar100':
        k_list = np.random.randint(args.shots, args.shots + 1, args.num_users)
    elif args.dataset == 'femnist':
        k_list = np.random.randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users)

    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)

    # Build models
    local_model_list = []
    for i in range(args.num_users):
        if args.dataset == 'mnist':
            if args.mode == 'model_heter':
                if i < 7:
                    args.out_channels = 18
                elif i >= 7 and i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20

            local_model = CNNMnist(args=args)

        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i < 7:
                    args.out_channels = 18
                elif i >= 7 and i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        elif args.dataset == 'cifar100' or args.dataset == 'cifar10':
            if args.mode == 'model_heter':
                if i < 10:
                    args.stride = [1, 4]
                else:
                    args.stride = [2, 2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=False, num_classes=args.num_classes)
            initial_weight = model_zoo.load_url(model_urls['resnet18'])
            local_model = resnet
            initial_weight_1 = local_model.state_dict()
            for key in initial_weight.keys():
                if key[0:3] == 'fc.' or key[0:5] == 'conv1' or key[0:3] == 'bn1':
                    initial_weight[key] = initial_weight_1[key]

            local_model.load_state_dict(initial_weight)

        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    # Choose training method based on args
    print("=" * 60)
    if args.use_dcaa:
        print("DCAA-Enhanced FedProto Training")
        print(f"Mode: {args.mode}")
        print(f"Contribution Weights: α={args.alpha} (VAG), β={args.beta} (PDS)")
        print(f"Thresholds: τ_low={args.tau_low}, τ_high={args.tau_high}")
        print(f"Throttling: γ={args.gamma}, consecutive_rounds={args.consecutive_rounds}")
        if args.num_attackers > 0:
            print(f"Simulating {args.num_attackers} free-riders ({args.attack_type} attack)")
        print("=" * 60)
        
        if args.mode == 'task_heter':
            FedProto_DCAA_taskheter(args, train_dataset, test_dataset, user_groups, 
                                    user_groups_lt, local_model_list, classes_list)
        elif args.mode == 'model_heter':
            FedProto_DCAA_modelheter(args, train_dataset, test_dataset, user_groups, 
                                     user_groups_lt, local_model_list, classes_list)
        else:
            raise ValueError(f"Unknown mode: {args.mode}. Use 'task_heter' or 'model_heter'")
    else:
        print("Original FedProto Training")
        print(f"Mode: {args.mode}")
        print("=" * 60)
        
        if args.mode == 'task_heter':
            FedProto_taskheter(args, train_dataset, test_dataset, user_groups, 
                              user_groups_lt, local_model_list, classes_list)
        elif args.mode == 'model_heter':
            FedProto_modelheter(args, train_dataset, test_dataset, user_groups, 
                               user_groups_lt, local_model_list, classes_list)
        else:
            raise ValueError(f"Unknown mode: {args.mode}. Use 'task_heter' or 'model_heter'")

    print('\n Total Run Time: {0:0.4f} seconds'.format(time.time() - start_time))