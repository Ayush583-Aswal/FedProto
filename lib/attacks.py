"""
Free-rider attack implementations for FedProto
"""

import torch
import numpy as np


class FreeRiderAttack:
    """Base class for free-rider attacks in FedProto"""
    
    def __init__(self, attack_type='plain'):
        self.attack_type = attack_type
    
    def generate_fake_protos(self, global_protos, global_protos_prev=None):
        """Generate fake prototypes based on attack type"""
        raise NotImplementedError


class PlainReplayAttack(FreeRiderAttack):
    """Simply returns the previous global prototypes"""
    
    def __init__(self):
        super().__init__(attack_type='plain')
    
    def generate_fake_protos(self, global_protos, global_protos_prev=None):
        if not global_protos:
            return {}
        return {k: v.clone().detach() for k, v in global_protos.items()}


class PerturbationAttack(FreeRiderAttack):
    """Adds Gaussian noise to previous global prototypes"""
    
    def __init__(self, sigma=0.1):
        super().__init__(attack_type='perturbation')
        self.sigma = sigma
    
    def generate_fake_protos(self, global_protos, global_protos_prev=None):
        if not global_protos:
            return {}
        
        fake_protos = {}
        for label, proto in global_protos.items():
            noise = torch.randn_like(proto) * self.sigma
            fake_protos[label] = (proto + noise).detach()
        return fake_protos


class ExtrapolationAttack(FreeRiderAttack):
    """Extrapolates based on prototype movement trajectory"""
    
    def __init__(self):
        super().__init__(attack_type='extrapolation')
    
    def generate_fake_protos(self, global_protos, global_protos_prev=None):
        if not global_protos:
            return {}
        
        if global_protos_prev is None or not global_protos_prev:
            # Fall back to plain replay if no history
            return {k: v.clone().detach() for k, v in global_protos.items()}
        
        fake_protos = {}
        for label in global_protos.keys():
            if label in global_protos_prev:
                velocity = global_protos[label] - global_protos_prev[label]
                fake_protos[label] = (global_protos[label] + velocity).detach()
            else:
                fake_protos[label] = global_protos[label].clone().detach()
        
        return fake_protos