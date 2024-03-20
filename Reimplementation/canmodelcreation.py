import owlready2 as owl
from owlready2 import *
import types
from owlready2 import get_ontology

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

        # add \Top
        
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


def get_canonical_model_elements(concept_names_iter, role_names_iter, ontology):
    
    onto = ontology
    top = owl.Thing
    bottom = owl.Nothing

    CanonicalModelElements(top)
    CanonicalModelElements(bottom)

    for concept_name in concept_names_iter:
        
        CanonicalModelElements(concept_name)
        for concept_name2 in concept_names_iter:
        
            with onto:
                gca = GeneralClassAxiom(concept_name & concept_name2)
                gca.is_a.append(concept_name & concept_name2)
            
            CanonicalModelElements(gca.left_side)

    print('')
    print('===========================================================================================================')
    print('All Concept Names and Concept Intersections have been preprocessed for the creation of the canonical model.')
    print('===========================================================================================================')

    concept_names_iter.append(top)
    #concept_names_iter.append(bottom)

    for role_name in role_names_iter:
        for concept_name in concept_names_iter:
            with onto:
                gca = GeneralClassAxiom(role_name.some(concept_name))
                gca.is_a.append(role_name.some(concept_name))

            CanonicalModelElements(gca.left_side)

    print('')
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

    def __init__(self, concept_names_dict, concept_intersections_dict, concept_restrictions_dict, all_concepts_dict, role_names_iter):
        
        self.domain = all_concepts_dict
        self.concept_names_dict = concept_names_dict
        self.concept_restrictions_dict = concept_restrictions_dict
        self.concept_intersections_dict = concept_intersections_dict

        self.role_names_iter = role_names_iter

        self.concept_canonical_interp = self.get_concept_name_caninterp() # These are only used to build the concept_canonical_interpretation and role_canonical_interpretation class attributes
        self.role_canonical_interp = self.get_role_name_caninterp()       # The functions do not return anything, they just update their corresponding class variables 

    def get_concept_name_caninterp(self):

        # The variable concept is a string containing the name of an element of the domain of the canonical model
        # The key to the concept_names_dict variable corresponds to concept.name
        # This name can be used to access the concept in owlready2's format

        for concept in self.concept_names_dict.keys():

            CanonicalModel.concept_canonical_interpretation[concept] = []
            superclasses = self.domain[concept].concept.ancestors(include_self=True, include_constructs=True) # The self.domain[concept] is used to access the CanonicalModelElements type of object,
                                                                                                               # and the attribute .concept is used to access the concept in owlready2 format
                                                                                                              
            for superclass in superclasses:

                if type(superclass) == ThingClass:
                    CanonicalModel.concept_canonical_interpretation[concept].append(superclass.name)

                elif type(superclass) == Restriction:
                    CanonicalModel.concept_canonical_interpretation[concept].append('exists_' + superclass.property.name + '.' + superclass.value.name)

                elif type(superclass) == And:
                    if 'And_' + ''.join(sorted(superclass.Classes[0].name + superclass.Classes[1].name)) in CanonicalModel.concept_canonical_interpretation[concept]:
                        pass
                    else:
                        CanonicalModel.concept_canonical_interpretation[concept].append('And_' + ''.join(sorted(superclass.Classes[0].name + superclass.Classes[1].name)))

            
    def get_role_name_caninterp(self):

        # Second case from Definition 10

        for role_name in self.role_names_iter:

            role_name_str = role_name.name # Accesses the property type object's name as a string
            CanonicalModel.role_canonical_interpretation[role_name_str] = []

            for restriction_name in self.concept_restrictions_dict.keys(): # Where restriction_name denotes a \exists r.B type of concept 'exists_' + self.concept.property.name + '.' + self.concept.value.name
                c_B = self.concept_restrictions_dict[restriction_name].concept.value.name

                if role_name_str in restriction_name:
                    
                    superclasses = self.domain[restriction_name].concept.ancestors(include_self=True, include_constructs=False) # Include_constructs is turned to false due to the definition of canonical model

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

        # First case from Definition 10

        for restriction_name in self.concept_restrictions_dict.keys():

            concept_name_str = self.concept_restrictions_dict[restriction_name].concept.value.name
            role_name_str = self.concept_restrictions_dict[restriction_name].concept.property.name

            role_name = self.concept_restrictions_dict[restriction_name].concept.property

            superroles = list(role_name.ancestors(include_self=True))
            
            pair = (restriction_name, concept_name_str)

            for superrole in superroles:
                if superrole.name in CanonicalModel.role_canonical_interpretation.keys():
                    CanonicalModel.role_canonical_interpretation[superrole.name].append(pair)
                else:
                    pass
'''
Main function for creating the canonical model.

    Args:
        onto_dir (str): a string pointing to the directory where the ontology is stored.

    Returns:
        canmodel (CanonicalModel): returns a variable containing the canonical model. 
        
        Attention: the interpretations of concept names and role names can also be accessed via class variables
        from the CanonicalModel class.
'''

def create_canonical_model(onto_dir):

    onto = get_ontology(onto_dir)
    onto = onto.load()

    individuals_iter = list(onto.individuals())
    gcas_iter = list(onto.general_class_axioms()) # Attention: this will not work unless the generator is converted into a list
    concept_names_iter = list(onto.classes())
    role_names_iter = list(onto.properties())
    individuals_iter = list(onto.individuals())

    # preprocess_namespaces(onto, individuals_iter, role_names_iter, concept_names_iter)


    get_canonical_model_elements(concept_names_iter, role_names_iter, onto)

    print('============================================================================')
    print('Starting to reason.\n')

    with onto:
        sync_reasoner()
        
    #onto.save("inferences_goslimyeast.owl")

    gcas_iter = list(onto.general_class_axioms()) # Attention: this will not work unless the generator is converted into a list
    concept_names_iter = list(onto.classes())
    role_names_iter = list(onto.properties())
    individuals_iter = list(onto.individuals())

    print('')
    print('============================================================================')
    print('Done reasoning. Creating the canonical model.')
    canmodel = CanonicalModel(CanonicalModelElements.concept_names, CanonicalModelElements.concept_intersections, CanonicalModelElements.concept_restrictions, CanonicalModelElements.all_concepts, role_names_iter)
    print('============================================================================\n')
    print('Concluded creating canonical model.')

    return canmodel


'''
Utility functions for initializing
the class EntityEmbedding. They
allow us to access dictionaries
containing indexes and canonical
interpretation of concepts
and roles as class.
'''

def get_concept_names_idx_dict(canmodel):
   conceptnames_idx_dict = {concept_name: idx for idx, concept_name in enumerate(CanonicalModel.concept_canonical_interpretation.keys())}
   return conceptnames_idx_dict

def get_role_names_idx_dict(canmodel):
    rolenames_idx_dict = {role_name: idx for idx, role_name in enumerate(CanonicalModel.role_canonical_interpretation.keys())}
    return rolenames_idx_dict

def get_entities_idx_dict(canmodel):
    entities_idx_dict = {entity: idx for idx, entity in enumerate(canmodel.domain.keys())}
    return entities_idx_dict

def get_domain_dict(canmodel):
    return canmodel.domain