import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, TensorDataset, ConcatDataset
from torch.utils.data import Dataset

import owlready2 as owl
from owlready2 import *
owlready2.reasoning.JAVA_MEMORY = 200000

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

from create_canonical_model import CanonicalModel, dir
from create_geometric_interpretation import idx_finder_dict, INCLUDE_TOP, EMB_DIM, SCALE_FACTOR, GeometricInterpretation, EntityEmbedding

torch.manual_seed(33)

TRAIN_SIZE_PROPORTION = 0.7 # Controls the proportion of training and test sets
BATCH_SIZE = 1 # Desired batch size

train_path = '/Users/victorlacerda/Documents/VSCode/ELH-Implementation-New/data/train_dataset.pt' # Path to save the training dataset
test_path = '/Users/victorlacerda/Documents/VSCode/ELH-Implementation-New/data/test_dataset.pt' # Path to save the test dataset

train_concept_loader_path = '/Users/victorlacerda/Documents/VSCode/ELH-Implementation-New/data/train_concept_loader.pt'
test_concept_loader_path = '/Users/victorlacerda/Documents/VSCode/ELH-Implementation-New/data/test_concept_loader.pt'

train_role_loader_path = '/Users/victorlacerda/Documents/VSCode/ELH-Implementation-New/data/train_role_loader.pt'
test_role_loader_path = '/Users/victorlacerda/Documents/VSCode/ELH-Implementation-New/data/train_test_loader.pt'

'''
Function for generating 
'''

def get_restriction_vertices(restriction_concept, concept_geointerps_dict, role_geointerps_dict, index_finder_dict, CanonicalModel: CanonicalModel, EntityEmbedding: EntityEmbedding, include_top_flag):
    
    concept_name = restriction_concept.value
    role_name = restriction_concept.property

    concept_name_str = concept_name.name
    role_name_str = role_name.name

    vertices = []

    for element in list(CanonicalModel.role_canonical_interpretation.keys()):
        if role_name_str in element:
            role_interpretation_set = CanonicalModel.role_canonical_interpretation[role_name_str]

            for pair in role_interpretation_set:
                element_1 = pair[0]
                element_2 = pair[1]

                if element_2 in CanonicalModel.concept_canonical_interpretation[concept_name_str]:
                    vertices.append(EntityEmbedding.entity_entityvector_dict[element_1])
    
    return np.array(vertices)

def get_intersection_vertices(restriction_concept, concept_geointerps_dict, role_geointerps_dict, index_finder_dict, CanonicalModel: CanonicalModel, EntityEmbedding: EntityEmbedding):
    pass

'''
Function for creating the pre-split dataset containing facts from the ontology.
Distinguishes between concept assertions and role assertions.


    Args: ontology_dir (str): the directory from the ontology
          concept_geointerps_dict (dict): the geometrical interpretations for concepts generated by create_tbox_embeddings()
          role_geointerps_dict (dict): the geometrical interpretations for roles generated by create_tbox_embeddings()
          concept_to_idx, role_to_idx (dict): A vocabulary {key (str): value (int)} from concept/role names to integer representation
          index_finder_dict (dict): Dictionary {key (int): value{tuple of strs}} storing information about the indices representing concept names
                                    and (role, entity) pairs for the embedding function mu. For example, {0: 'Thing'} and {322: ('P22', 'exists_P22.Child')}.
          emb_dim (int): integer representing the embedding dimension.
          CanonicalModel (CanonicalModel): Data class for storing the canonical model's domain and interpretations.
          EntityEmbedding (EntityEmbedding): Data class for storing the embedded elements of the canonical model's domain.

    Returns:
          X_concepts (np.array): A dataframe with columns 'Concept', 'Entity', 'y_true' (equivalent to concept.centroid())
          X_roles (np.array): A dataframe with columns 'SubjectEntity', 'Role', 'ObjectEntity', 'y_true' (equivalent to role.centroid())
          y_concepts (np.array):
          y_roles (np.array):
          vocabulary_dict (dict): A vocabulary with key (int): value (str) for entities in the domain.
'''

def get_abox_dataset(ontology_dir: str,
                     concept_geointerps_dict: dict, role_geointerps_dict: dict,
                     concept_to_idx: dict, role_to_idx: dict,
                     index_finder_dict: dict, emb_dim: int,
                     CanonicalModel: CanonicalModel, EntityEmbedding: EntityEmbedding,
                     include_top_flag: bool):
    
    ontology = get_ontology(ontology_dir)
    ontology = ontology.load()

    X_concepts = []
    X_roles = []
    y_concepts = []
    y_roles = []

    entities = [entity.name for entity in list(ontology.individuals())]
    
    if include_top_flag == True:
        concept_to_idx_vocab = concept_to_idx.copy()
        concept_to_idx_vocab = {k: v+1 for k,v in concept_to_idx_vocab.items()}
        concept_to_idx_vocab.update({'Thing': 0})
    else:
        concept_to_idx_vocab = concept_to_idx

    idx_to_concept_vocab = {value: key for key, value in concept_to_idx_vocab.items()}

    role_to_idx_vocab = role_to_idx
    idx_to_role_vocab = {value: key for key, value in role_to_idx_vocab.items()}
    
    entity_to_idx_vocab = {value: index for index, value in enumerate(entities)}
    idx_to_entity_vocab = {value: key for key, value in entity_to_idx_vocab.items()}

    for individual in list(ontology.individuals()):

        if include_top_flag == True:
            all_facts = individual.INDIRECT_is_a
        else:
            preprocessed_all_facts = individual.INDIRECT_is_a
            all_facts = []
            
            for fact in preprocessed_all_facts:
                if type(fact) == ThingClass and fact.name == 'Thing':
                    pass
                elif type(fact) == And and (fact.Classes[0].name == 'Thing' or fact.Classes[1].name == 'Thing'):
                    pass
                elif type(fact) == Restriction and fact.value.name == 'Thing':
                    pass
                else:
                    all_facts.append(fact)

        for concept in all_facts:
            # Handles concepts of the type A

            if type(concept) == ThingClass:
                concept = concept_geointerps_dict[concept.name]
                fact = np.array([concept_to_idx_vocab[concept.name], entity_to_idx_vocab[individual.name]])
                y_label = np.array(concept.centroid)
                X_concepts.append(fact)
                y_concepts.append(y_label)
                
            # Handles concepts of the type A \and B
            elif type(concept) == And:
                concept1 = concept_geointerps_dict[concept.Classes[0]]
                concept2 = concept_geointerps_dict[concept.Classes[1]]
                intersection_name = 'And_' + ''.join(sorted(concept1.name + concept2.name))

                if concept_to_idx_vocab.get(intersection_name) is not None:
                    fact = np.array([concept_to_idx_vocab[intersection_name], entity_to_idx_vocab[individual.name]])
                    y_label = np.array((concept1.centroid + concept2.centroid)/2) # The golden label for an intersection is just the average of the centroid of the two regions
                    X_concepts.append(fact)
                    y_concepts.append(y_label)

                else:
                    concept_to_idx_vocab[intersection_name] = len(concept_to_idx_vocab)
                    idx_to_concept_vocab[len(concept_to_idx_vocab)] = intersection_name
                    fact = np.array([concept_to_idx_vocab[intersection_name], entity_to_idx_vocab[individual.name]])
                    y_label = np.array((concept1.centroid + concept2.centroid)/2) # The golden label for an intersection is just the average of the centroid of the two regions
                    X_concepts.append(fact)
                    y_concepts.append(y_label)

            # Handles concepts of the type \exists r.B
            elif type(concept) == Restriction:
                concept_name = concept.value
                role_name = concept.property
                restriction_name = 'exists_' + role_name.name + '.' + concept_name.name

                if concept_to_idx_vocab.get(restriction_name) is not None:
                
                    fact = np.array([concept_to_idx_vocab[restriction_name], entity_to_idx_vocab[individual.name]])
                    y_label = np.array(GeometricInterpretation.concept_geointerps_dict[restriction_name].centroid)
                    X_concepts.append(fact)
                    y_concepts.append(y_label)

                else:
                    concept_to_idx_vocab[restriction_name] = len(concept_to_idx_vocab)
                    idx_to_concept_vocab[len(concept_to_idx_vocab)-1] = restriction_name
                    restriction_concept = GeometricInterpretation(restriction_name, EMB_DIM) # Initializes a Geometric Interpretation type object
                    restriction_concept.vertices = get_restriction_vertices(concept, concept_geointerps_dict,
                                                                            role_geointerps_dict, index_finder_dict, CanonicalModel, EntityEmbedding,
                                                                            include_top_flag)
                    
                    GeometricInterpretation.concept_geointerps_dict[restriction_name] = restriction_concept
                    restriction_concept.centroid = restriction_concept.get_centroid_naive()
                    fact = np.array([concept_to_idx_vocab[restriction_concept.name], entity_to_idx_vocab[individual.name]])
                    y_label = restriction_concept.centroid
                    X_concepts.append(fact)
                    y_concepts.append(y_label)

        relevant_roles = individual.get_properties()
        individual_name = individual.name

        for role in relevant_roles:

            role_geo = role_geointerps_dict[role.name]
            subject_list = role[individual] # This syntax is from the owlready2 library

            for subject in subject_list:
                fact = np.array([entity_to_idx_vocab[individual.name], role_to_idx_vocab[role.name], entity_to_idx_vocab[subject.name]])

                X_roles.append(fact)
                y_label = y_roles.append(np.array(role_geo.centroid))

    return np.array(X_concepts), np.array(X_roles), np.array(y_concepts), np.array(y_roles), entity_to_idx_vocab, idx_to_entity_vocab, concept_to_idx_vocab, idx_to_concept_vocab, role_to_idx_vocab, idx_to_role_vocab

X_concepts, X_roles, y_concepts, y_roles, entity_to_idx_vocab, idx_to_entity_vocab, concept_to_idx_vocab, idx_to_concept_vocab, role_to_idx_vocab, idx_to_role_vocab = get_abox_dataset(dir,
                                                                                                                                                                                        GeometricInterpretation.concept_geointerps_dict,
                                                                                                                                                                                        GeometricInterpretation.role_geointerps_dict,
                                                                                                                                                                                        EntityEmbedding.concept_names_idx_dict,
                                                                                                                                                                                        EntityEmbedding.role_names_idx_dict,
                                                                                                                                                                                        idx_finder_dict,
                                                                                                                                                                                        EMB_DIM,
                                                                                                                                                                                        CanonicalModel, EntityEmbedding,
                                                                                                                                                                                        INCLUDE_TOP
                                                                                                                                                                                        )



class OntologyDataset(Dataset):
    def __init__(self, data, labels):
        self.X = torch.tensor(data, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].long(), self.y[idx]
    
    pass


ConceptDataset = OntologyDataset(X_concepts, y_concepts)
dataset_size = len(ConceptDataset)
train_size = int(TRAIN_SIZE_PROPORTION * dataset_size)
test_size = dataset_size - train_size
trainConceptDataset, testConceptDataset = torch.utils.data.random_split(ConceptDataset, [train_size, test_size])

RoleDataset = OntologyDataset(X_roles, y_roles)
dataset_size = len(RoleDataset)
train_size = int(TRAIN_SIZE_PROPORTION * dataset_size)
test_size = dataset_size - train_size
trainRoleDataset, testRoleDataset = torch.utils.data.random_split(RoleDataset, [train_size, test_size])

ConceptDataLoader = DataLoader(ConceptDataset, batch_size = BATCH_SIZE, shuffle=True)
train_ConceptDataLoader = DataLoader(trainConceptDataset, batch_size = BATCH_SIZE, shuffle=True)
test_ConceptDataLoader = DataLoader(testConceptDataset, batch_size = BATCH_SIZE, shuffle=True)

RoleDataLoader = DataLoader(RoleDataset, batch_size = BATCH_SIZE, shuffle=True)
train_RoleDataLoader = DataLoader(trainRoleDataset, batch_size = BATCH_SIZE, shuffle=True)
test_RoleDataLoader = DataLoader(testRoleDataset, batch_size = BATCH_SIZE, shuffle=True)

# Save train dataset
torch.save(trainRoleDataset, train_path)

# Save test dataset
torch.save(testRoleDataset, test_path)

torch.save(train_ConceptDataLoader, train_concept_loader_path)
torch.save(test_ConceptDataLoader, test_concept_loader_path)

torch.save(train_RoleDataLoader, train_role_loader_path)
torch.save(test_RoleDataLoader, test_role_loader_path)