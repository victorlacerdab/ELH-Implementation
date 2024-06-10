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

#dir = '/Users/victorlacerda/Documents/VSCode/ELHFaithfulness/NormalizedOntologies/family_ontology.owl'
dir = '/Users/victorlacerda/Desktop/family_ontology.owl'
#dir = '/Users/victorlacerda/Desktop/TestRole.owl'

RESTRICT_LANGUAGE = False # If True, the language is restricted to simpler TBox axioms on the left-hand side of the rules
INCLUDE_TOP = True

'''
Class for creating entities to
populate the creation of the
canonical models.
The .name attribute is used to
create a single representation
for concepts like A and B / 
B and A, as they are the same.
'''

class CanonicalModelElements:

    concept_names = {}
    concept_intersections = {}
    concept_restrictions = {}
    all_concepts = {}

    def __init__(self, concept):
        self.concept = concept
        self.name = self.get_name()
        self.get_element_dict()

    def get_name(self):
        
        if type(self.concept) == ThingClass:
            return self.concept.name

        elif type(self.concept) == Restriction:
            return 'exists_' + self.concept.property.name + '.' + self.concept.value.name
        
        else:
            return 'And_' + ''.join(sorted(self.concept.Classes[0].name + self.concept.Classes[1].name)) # The name is sorted to avoid that (e.g) (A \and B) and (B \and A) are treated as different concepts
        
    def get_element_dict(self):

        if type(self.concept) == ThingClass:
            CanonicalModelElements.concept_names[self.name] = self
            CanonicalModelElements.all_concepts[self.name] = self

        elif type(self.concept) == Restriction:
            CanonicalModelElements.concept_restrictions[self.name] = self
            CanonicalModelElements.all_concepts[self.name] = self

        elif type(self.concept) == And:
            CanonicalModelElements.concept_intersections[self.name] = self
            CanonicalModelElements.all_concepts[self.name] = self

def get_canonical_model_elements(concept_names_iter, role_names_iter, ontology, restrict_language = False, include_top = True):
    
    onto = ontology

    if include_top == True:
        top = owl.Thing
        CanonicalModelElements(top)
        #bottom = owl.Nothing
        #CanonicalModelElements(bottom)

    for concept_name in concept_names_iter:
        CanonicalModelElements(concept_name)

        if restrict_language == False:

            for concept_name2 in concept_names_iter:
        
                with onto:
                    gca = GeneralClassAxiom(concept_name & concept_name2)
                    gca.is_a.append(concept_name & concept_name2)
            
                CanonicalModelElements(gca.left_side)

    print('')
    print('===========================================================================================================')
    print('All Concept Names and Concept Intersections have been preprocessed for the creation of the canonical model.')
    print('===========================================================================================================')

    if include_top == True:
        concept_names_iter.append(top)
        #concept_names_iter.append(bottom)
    else:
        print('Top is not being included in the canonical model.')
        pass

    if restrict_language == False:
        for role_name in role_names_iter:
            for concept_name in concept_names_iter:
                with onto:
                    gca = GeneralClassAxiom(role_name.some(concept_name))
                    gca.is_a.append(role_name.some(concept_name))

                CanonicalModelElements(gca.left_side)
    
    else:
        if include_top == True:
            for role_name in role_names_iter:
                with onto:
                    gca = GeneralClassAxiom(role_name.some(owl.Thing))
                    gca.is_a.append(role_name.some(owl.Thing))
                    CanonicalModelElements(gca.left_side)
        else:
            print(f'Top must be included when restricting the language. Terminating the creation of the canonical model.')
            sys.exit(1)
            
    print('')
    print('All restrictions have been preprocessed for the creation of the canonical model.')


'''
The main class for creating the canonical model for the ontology.

The canonical model is stored in dictionaries available as class variables 'concept_canonical_interpretation'
and 'role_canonical_interpretation'. 

Args:
    concept_names_dict: a dictionary stored in the CanonicalModelElement class.
    concept_intersection_dict: a dictionary stored in the CanonicalModelElement class.
    concept_restrictions_dict: a dictionary stored in the CanonicalModelElement class.
    all_concepts_dict: a dictionary stored in the CanonicalModelElement class.
    role_names_iter (list): a list containing all role names in the loaded ontology.
'''

class CanonicalModel:

    concept_canonical_interpretation = {}
    role_canonical_interpretation = {}

    def __init__(self, concept_names_dict, concept_intersections_dict, concept_restrictions_dict, all_concepts_dict, role_names_iter, include_top_flag):
        
        self.domain = all_concepts_dict
        self.concept_names_dict = concept_names_dict
        self.concept_restrictions_dict = concept_restrictions_dict
        self.concept_intersections_dict = concept_intersections_dict
        self.include_top = include_top_flag

        self.role_names_iter = role_names_iter

        self.concept_canonical_interp = self.get_concept_name_caninterp() # These are only used to build the concept_canonical_interpretation and role_canonical_interpretation class attributes
        self.role_canonical_interp = self.get_role_name_caninterp()       # The functions do not return anything, they just update their corresponding class variables 

    def get_concept_name_caninterp(self):

        # The variable concept is a string containing the name of an element of the domain of the canonical model
        # The key to the concept_names_dict variable corresponds to concept.name
        # This name can be used to access the concept in owlready2's format

        for concept in self.concept_names_dict.keys():

            CanonicalModel.concept_canonical_interpretation[concept] = []
            if self.include_top == True:
                superclasses = self.domain[concept].concept.ancestors(include_self=True, include_constructs=True) # The self.domain[concept] is used to access the CanonicalModelElements type of object,
                                                                                                                  # and the attribute .concept is used to access the concept in owlready2 format

            else:
                superclasses = list(self.domain[concept].concept.ancestors(include_self=True, include_constructs=True))
                superclasses = [superclass for superclass in superclasses if type(superclass) == ThingClass and superclass.name != 'Thing']

            for superclass in superclasses:

                if type(superclass) == ThingClass:
                    CanonicalModel.concept_canonical_interpretation[concept].append(superclass.name)
                    if superclass != owl.Thing:
                        CanonicalModel.concept_canonical_interpretation[concept].append('And_' + ''.join(sorted(superclass.name + superclass.name)))

                elif type(superclass) == Restriction:
                    CanonicalModel.concept_canonical_interpretation[concept].append('exists_' + superclass.property.name + '.' + superclass.value.name)

                elif type(superclass) == And:
                    if 'And_' + ''.join(sorted(superclass.Classes[0].name + superclass.Classes[1].name)) in CanonicalModel.concept_canonical_interpretation[concept]:
                        pass
                    else:
                        CanonicalModel.concept_canonical_interpretation[concept].append('And_' + ''.join(sorted(superclass.Classes[0].name + superclass.Classes[1].name)))

            
    def get_role_name_caninterp(self):

        # Initialize the dictionary storing the canonical interpretation of roles

        for role_name in self.role_names_iter:

            role_name_str = role_name.name # Accesses the property type object's name as a string
            CanonicalModel.role_canonical_interpretation[role_name_str] = []

        # First case from Definition 10
                                
        for role_name in self.role_names_iter:

            superroles = role_name.ancestors(include_self=True)

            for superrole in superroles:
                for restriction_name, restriction_concept in self.concept_restrictions_dict.items():

                    if superrole == restriction_concept.concept.property:
                        concept_name_str = restriction_concept.concept.value.name
                        pair = (restriction_name, concept_name_str)
                        CanonicalModel.role_canonical_interpretation[role_name.name].append(pair)
                        
        # Second case from Definition 10

        for restriction_name in self.concept_restrictions_dict.keys(): # Where restriction_name denotes an \exists r.B type of concept 'exists_' + self.concept.property.name + '.' + self.concept.value.name

            # print(f'Restriction name init for loop: {restriction_name}')
            restriction_concept = self.concept_restrictions_dict[restriction_name].concept
            c_B = self.concept_restrictions_dict[restriction_name].concept.value.name
            # print(f'c_B: {c_B}\n')

            superclasses = restriction_concept.ancestors(include_self=True, include_constructs=False)

            for superclass in superclasses:

                super_superclasses = superclass.ancestors(include_self=True, include_constructs=True)

                for super_superclass in super_superclasses:
                    if type(super_superclass) == ThingClass:
                        c_D = super_superclass.name
                        CanonicalModel.role_canonical_interpretation[role_name_str].append((c_D, c_B))

                    elif type(super_superclass) == Restriction:
                        c_D = 'exists_' + super_superclass.property.name + '.' + super_superclass.value.name
                        CanonicalModel.role_canonical_interpretation[role_name_str].append((c_D, c_B))

                    elif type(super_superclass) == And:
                        c_D = 'And_' + ''.join(sorted(super_superclass.Classes[0].name + super_superclass.Classes[1].name))
                        CanonicalModel.role_canonical_interpretation[role_name_str].append((c_D, c_B))
                    
            if role_name_str in restriction_name:

                #print(f'It is true that {role_name_str} is in {restriction_name}.')
                    
                superclasses = self.domain[restriction_name].concept.ancestors(include_self=True, include_constructs=False) # Include_constructs is turned to false due to the definition of canonical model

                # print(f'These are the superclasses of the restriction_name {restriction_name}:" {superclasses}')

                for superclass in superclasses:
                    super_superclasses = superclass.ancestors(include_self=True, include_constructs=True)

                    for super_superclass in super_superclasses:

                        if type(super_superclass) == ThingClass:
                            c_D = super_superclass.name
                            CanonicalModel.role_canonical_interpretation[role_name_str].append((c_D, c_B))

                        elif type(super_superclass) == Restriction:
                            c_D = 'exists_' + super_superclass.property.name + '.' + super_superclass.value.name
                            CanonicalModel.role_canonical_interpretation[role_name_str].append((c_D, c_B))

                        elif type(super_superclass) == And:
                            c_D = 'And_' + ''.join(sorted(super_superclass.Classes[0].name + super_superclass.Classes[1].name))
                            CanonicalModel.role_canonical_interpretation[role_name_str].append((c_D, c_B))

'''
Main function for creating the canonical model.

    Args:
        onto_dir (str): a string pointing to the directory where the ontology is stored.

    Returns:
        canmodel (CanonicalModel): returns a variable containing the canonical model. 
        
        Attention: the interpretations of concept names and role names can also be accessed via class variables
        from the CanonicalModel class.
'''

def create_canonical_model(onto_dir: str, restrict_language_flag: bool, include_top_flag: bool):

    onto = get_ontology(onto_dir)
    onto = onto.load()

    individuals_iter = list(onto.individuals())
    gcas_iter = list(onto.general_class_axioms()) # Attention: this will not work unless the generator is converted into a list
    concept_names_iter = list(onto.classes())
    role_names_iter = list(onto.properties())

    get_canonical_model_elements(concept_names_iter, role_names_iter, onto, restrict_language_flag, include_top_flag)

    print('============================================================================')
    print('Starting to reason.\n')

    with onto:
        sync_reasoner()

    gcas_iter = list(onto.general_class_axioms()) # Attention: this will not work unless the generator is converted into a list
    concept_names_iter = list(onto.classes())
    role_names_iter = list(onto.properties())
    individuals_iter = list(onto.individuals())

    print('')
    print('============================================================================')
    print('Done reasoning. Creating the canonical model.')
    canmodel = CanonicalModel(CanonicalModelElements.concept_names, CanonicalModelElements.concept_intersections, CanonicalModelElements.concept_restrictions, CanonicalModelElements.all_concepts, role_names_iter, INCLUDE_TOP)
    print('============================================================================\n')
    print('Concluded creating canonical model.')

    return canmodel

# Instantiates the canonical model

canmodel = create_canonical_model(dir, RESTRICT_LANGUAGE, INCLUDE_TOP)