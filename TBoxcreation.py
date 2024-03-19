import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import owlready2 as owl
from owlready2 import *
import types
owlready2.reasoning.JAVA_MEMORY = 200000

#import scipy
import pandas as pd
import numpy as np

from canmodelcreation import get_concept_names_idx_dict, get_role_names_idx_dict, get_entities_idx_dict, get_domain_dict

''' 
Class for obtaining the positional 
embedding for each entity in the domain
of the canonical interpretation.
It represents the Mu Function from the
paper.
'''

class EntityEmbedding:

    # Dictionaries for storing the indices of concept names and role names, entities pairs, respectively
    # Keys are strings and values are integers
    
    concept_names_idx_dict = get_concept_names_idx_dict(canmodel)
    role_names_idx_dict = get_role_names_idx_dict(canmodel)
    entities_idx_dict = get_entities_idx_dict(canmodel)

    # Dictionaries accessing the canonical interpretation of concepts and roles
    # Keys and values are strings
    
    concept_canonical_interpretation_dict = CanonicalModel.concept_canonical_interpretation
    role_canonical_interpretation_dict = CanonicalModel.role_canonical_interpretation

    # Dictionary storing the domain of the canonical model being embedded
    # IMPORTANT: Keys are strings and values are CanonicalModelElements type objects
    
    domain_dict = get_domain_dict(canmodel)

    # Dictionary for easy access to entity embeddings
    # It is initialized with empty values, iteratively built by the .get_embedding_vector() method
    # Key (str): Domain Entity / Value (np.array): EntityEmbedding.embedding_vector

    entity_entityvector_dict = dict.fromkeys(domain_dict.keys())

    def __init__(self, entity_name, emb_dim, scale_factor):
        self.name = entity_name
        self.emb_dim = emb_dim
        self.scale_factor = scale_factor
        self.in_interpretation_of = []
        self.embedding_vector = self.get_embedding_vector()

    def get_embedding_vector(self):
        
        embedding_vector = np.zeros((self.emb_dim,))
        EntityEmbedding.entity_entityvector_dict[self.name] = []

        # Applies the embedding function to the concept names portion of the definition

        for concept_name in EntityEmbedding.concept_canonical_interpretation_dict:
            concept_name_idx = EntityEmbedding.concept_names_idx_dict[concept_name]
        
            if self.name in EntityEmbedding.concept_canonical_interpretation_dict[concept_name]:
                embedding_vector[concept_name_idx] = 1 * self.scale_factor
                self.in_interpretation_of.append(concept_name)

        # Applies the embedding function to the role names portion of the definition

        for role_name in EntityEmbedding.role_canonical_interpretation_dict:
            
            role_name_idx = len(EntityEmbedding.concept_names_idx_dict) + (EntityEmbedding.role_names_idx_dict[role_name] * len(EntityEmbedding.entities_idx_dict))
            role_name_caninterp = EntityEmbedding.role_canonical_interpretation_dict[role_name]

            for pair in role_name_caninterp:

                entity_2 = pair[1]

                if (self.name, entity_2) == pair:
                    entity_2_idx = EntityEmbedding.entities_idx_dict[entity_2]
                    final_role_entity_pair_idx = role_name_idx + entity_2_idx
                    embedding_vector[final_role_entity_pair_idx] = 1 * self.scale_factor

        # EntityEmbedding.entity_entityvector_dict[self.name].append(embedding_vector)
        EntityEmbedding.entity_entityvector_dict[self.name] = embedding_vector

        return embedding_vector
    

'''
Function for creating the binary vectors representing
each element of the domain of the canonical interpretation.

    Args:
        emb_dim (int/float): the number of dimensions of the embedding space.

    Returns:
        embedded_entities (list): a list containing all embeddings of the entities
                                  in the domain. 
    
    The embedded_entities are also available in the dictionary EntityEmbeddings.entity_entityvector_dict
'''

def get_domain_embeddings(emb_dim, scale_factor):

    embedded_entities = []
    counter = 0

   # The entities in the domain are strings
    
    for entity_name in EntityEmbedding.domain_dict:
       embedded_entity = EntityEmbedding(entity_name, emb_dim, scale_factor)
       embedded_entities.append(embedded_entity)
       counter += 1
       
       if counter % 1000 == 0:
           print(counter)
       
    return embedded_entities

'''
Final class for creating the dataset.

Inputs: concept or role names, generated
embeddings for entities in the domain of
the canonical model.

Outputs: geometrical interpretation of
concepts and role names, represented
by vertices defining a region.

One can access the GeometricInterpretation
objects either as elements in a list, or as
values in a class variable dictionary.
'''

class GeometricInterpretation:

    concept_geointerps_dict = dict.fromkeys(CanonicalModel.concept_canonical_interpretation.keys())
    role_geointerps_dict = dict.fromkeys(CanonicalModel.role_canonical_interpretation.keys())

    def __init__(self, name, emb_dim):
        self.name = name
        self.emb_dim = emb_dim
        self.vertices = []
        self.centroid = None

        if GeometricInterpretation.concept_geointerps_dict.get(name) is not None:
            GeometricInterpretation.concept_geointerps_dict[name] = []

    def get_centroid_naive(self):
        if len(self.vertices) == 0 and self.name in self.concept_geointerps_dict.keys():
            centroid = np.zeros((self.emb_dim,))
            return centroid
        
        elif len(self.vertices) == 0 and self.name in self.role_geointerps_dict.keys():
            centroid = np.zeros((self.emb_dim * 2,)) # The centroid for the regions needs to be doubled due to the concat operation
            return centroid
        
        elif len(self.vertices) > 0 and self.name in self.concept_geointerps_dict.keys():
            n = len(self.vertices)
            centroid = np.zeros((self.emb_dim,))
            matrix = np.vstack(self.vertices)
            centroid = 1/n * np.sum(matrix, axis=0)
            return centroid
        
        elif len(self.vertices) > 0 and self.name in self.role_geointerps_dict.keys():
            n = len(self.vertices)
            centroid = np.zeros((self.emb_dim,))
            matrix = np.vstack(self.vertices)
            centroid = 1/n * np.sum(matrix, axis=0)
            return centroid
        
def index_finder(emb_dim, concept_name_idx_dict, role_name_idx_dict, domain_idx_dict):

    index_dict = {k: None for k in range(emb_dim)}

    for k,v in concept_name_idx_dict.items():

        index_dict[v] = k

    for role in role_name_idx_dict:
        role_init_idx = len(concept_name_idx_dict) + (role_name_idx_dict[role] * len(domain_idx_dict))

        for entity in domain_idx_dict:
            entity_init_idx = domain_idx_dict[entity]
            final_role_entity_pair_idx = role_init_idx + entity_init_idx
            index_dict[final_role_entity_pair_idx] = (role, entity)
        
    return index_dict

def get_faithful_concept_geometric_interps(concept_names_interps, domain_embeddings_list, entity_dims_index_dict, emb_dim, canmodel: CanonicalModel):

    faithful_concept_geometric_interps = []

    for concept_name in concept_names_interps.keys():
        concept_name = GeometricInterpretation(concept_name, emb_dim)

        for embedding in domain_embeddings_list:
            if concept_name.name in embedding.in_interpretation_of:
                concept_name.vertices.append(embedding.embedding_vector)
            
        GeometricInterpretation.concept_geointerps_dict[concept_name.name] = concept_name
        concept_name.centroid = concept_name.get_centroid_naive()
        
        faithful_concept_geometric_interps.append(concept_name)

    return faithful_concept_geometric_interps

def get_faithful_role_geometric_interps(role_names_interps, entity_embeddings_list, entity_dims_index_dict, emb_dim, canmodel: CanonicalModel):
    
    faithful_role_geometric_interps = []
    idx_entity_dict = entity_dims_index_dict
    #entity_idx_dict = {v: k for k,v in entity_dims_index_dict}

    relevant_idxs = len(canmodel.concept_names_dict)-1

    for role_name in role_names_interps.keys():
        role_name_str = role_name
        role_name = GeometricInterpretation(role_name_str, emb_dim)

        for entity in entity_embeddings_list:

            onehot_idx_list = np.where(entity.embedding_vector == 1)[0]
            #print(f'This is the entity: {entity} and this is the onehot_idx_list: {onehot_idx_list}')

            for idx in onehot_idx_list: # I could just look at the TRULY relevant indexes
                if idx > relevant_idxs:
                    role_entity_pair = idx_entity_dict[idx]
                    r_name_str = role_entity_pair[0]
                    e_name_str = role_entity_pair[1]

                    if r_name_str == role_name_str:
                        e_embedding = EntityEmbedding.entity_entityvector_dict[e_name_str]
                        role_name.vertices.append(np.concatenate((entity.embedding_vector, e_embedding)))

        GeometricInterpretation.role_geointerps_dict[role_name_str] = role_name
        role_name.centroid = role_name.get_centroid_naive()
        faithful_role_geometric_interps.append(role_name)

    return faithful_role_geometric_interps