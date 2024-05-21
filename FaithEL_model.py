import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import owlready2 as owl
from owlready2 import *
owlready2.reasoning.JAVA_MEMORY = 200000

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from create_geometric_interpretation import SCALE_FACTOR, GeometricInterpretation

class FaithEL(nn.Module):
    def __init__(self, emb_dim, phi, radius, gamma,
                 individual_vocabulary,
                 concept_vocabulary,
                 role_vocabulary,
                 ):
        
        super(FaithEL, self).__init__()
        self.emb_dim = emb_dim
        self.phi = phi # Weights how much the concept/role parameter moves
        self.gamma = gamma # Weights how much the individual embeddings moves
        self.radius = radius

        self.individual_embedding_dict = nn.Embedding(len(individual_vocabulary),
                                                      emb_dim
                                                      )
        with torch.no_grad():
            value = SCALE_FACTOR/2
            std_dev = 0.3
            self.individual_embedding_dict.weight.data.normal_(mean=value, std=std_dev)
        
        self.concept_embedding_dict = nn.Embedding(len(concept_vocabulary),
                                                   emb_dim
                                                   )
        
        # Initializes the moving parameter for concepts at the concept's respective centroid
        
        with torch.no_grad():
            for concept_name, concept_idx in concept_vocabulary.items():
                self.concept_embedding_dict.weight[concept_idx] = torch.tensor(GeometricInterpretation.concept_geointerps_dict[concept_name].centroid)

        self.role_embedding_dict = nn.Embedding(len(role_vocabulary),
                                                emb_dim * 2
                                                )
        
        # Initializes the moving parameter for roles at the role's respective centroid
        with torch.no_grad():
            for role_name, role_idx in role_vocabulary.items():
                self.role_embedding_dict.weight[role_idx] = torch.tensor(GeometricInterpretation.role_geointerps_dict[role_name].centroid)
    
    def forward(self, data):
    
        # Concept assertions are of the form ['Concept', 'Entity']
        # Role assertions are of the form ['SubjectEntity', 'Role', 'ObjectEntity']
        
        subj_entity_idx = 1 if len(data[0]) == 2 else 0 # Checks whether the model has received a C assert or R assert

        if subj_entity_idx == 1:

            concept_idx = 0
            
            concept = data[:, concept_idx]
    
            subj_entity = data[:, subj_entity_idx]

            neg_object_entity = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (subj_entity.shape))

            out1 = self.concept_embedding_dict(concept) 
            out2 = self.individual_embedding_dict(subj_entity) 
            out3 = self.individual_embedding_dict(neg_object_entity) 

            return out1, out2, out3

        elif subj_entity_idx == 0:

            role_idx = 1
            obj_entity_idx = 2
            
            role = data[:, role_idx]

            subject_entity = data[:, subj_entity_idx]
            object_entity = data[:, obj_entity_idx]
            neg_object_entity = torch.randint(0, self.individual_embedding_dict.weight.shape[0], (subject_entity.shape))

            out1 = self.role_embedding_dict(role) # Role parameter embedding
            out2 = torch.cat((self.individual_embedding_dict(subject_entity), self.individual_embedding_dict(object_entity)), dim=1)
            out3 = torch.cat((self.individual_embedding_dict(subject_entity), self.individual_embedding_dict(neg_object_entity)), dim=1)
            
            return out1, out2, out3
        
    def concept_parameter_constraint(self):
        with torch.no_grad():
            for idx, weight in enumerate(self.concept_embedding_dict.weight):
                centroid = torch.tensor(GeometricInterpretation.concept_geointerps_dict[list(GeometricInterpretation.concept_geointerps_dict.keys())[idx]].centroid)
                distance = torch.dist(weight, centroid, p=2)
                if distance > self.radius:
                    self.concept_embedding_dict.weight[idx] = centroid + self.radius * (weight - centroid) / distance

    def role_parameter_constraint(self):
        with torch.no_grad():
            for idx, weight in enumerate(self.role_embedding_dict.weight):
                centroid = torch.tensor(GeometricInterpretation.role_geointerps_dict[list(GeometricInterpretation.role_geointerps_dict.keys())[idx]].centroid)
                distance = torch.dist(weight, centroid, p=2)
                if distance > self.radius:
                    self.role_embedding_dict.weight[idx] = centroid + self.radius * (weight - centroid) / distance