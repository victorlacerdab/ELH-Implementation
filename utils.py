import torch
from torch import nn, optim
import torch.nn.functional as F

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

'''
Computes hits@k for concept assertions.
'''

def get_hits_at_k_concept_assertions(model, GeoInterp_dataclass,
                  test_concept_assertions, test_role_assertions,
                  entity_to_idx_vocab: dict, idx_to_entity_vocab: dict,
                  idx_to_concept_vocab: dict, idx_to_role_vocab: dict,
                  centroid_score: False,
                  ):
    
    top1 = 0
    top3 = 0
    top10 = 0
    top100 = 0
    top_all = 0

    model.eval()

    hits = []

    relevant_concept_idx = []

    # Gathers only concepts appearing in the test set (it is not guaranteed that if a concept appears in the dataset, then it appears here)

    for assertion in test_concept_assertions:
        inputs, _ = assertion
        if inputs[0] not in relevant_concept_idx:
            relevant_concept_idx.append(inputs[0])
        else:
            pass

    # print(f'Relevant concept idx: {relevant_concept_idx}')

    with torch.no_grad():

        for concept_idx in relevant_concept_idx:

            assertion_scores = []

            for _, entity_idx in entity_to_idx_vocab.items():
                eval_sample = torch.tensor([concept_idx, entity_idx]).unsqueeze(0)
                outputs1, outputs2, outputs3 = model(eval_sample) # out1 = Concept parameter, out2 = Individual parameter

                if centroid_score == False:
                    assertion_score = torch.dist(outputs1, outputs2, p=2) # Distance from the individual embedding from the concept parameter embedding
                else:
                    assertion_score = torch.dist(outputs1, outputs2, p=2) + torch.dist(outputs2, torch.tensor(GeoInterp_dataclass.concept_geointerps_dict[idx_to_concept_vocab[int(concept_idx)]].centroid)) 
                    # Distance from the individual embedding from concept param embedding plus distance from the ind emb to the centroid of the geointerp of the concept

                assertion_scores.append((torch.tensor([concept_idx, entity_idx]), assertion_score.item()))
            
            sorted_scores = sorted(assertion_scores, key=lambda x: x[1])

            #print(f'Current query concept: {concept_idx}')
            #print(f'Centroid_score = {centroid_score}')
            #print(f'Assertion scores: {assertion_scores}')
            #print(f'Sorted scores: {sorted_scores}\n')

            k_list = [1, 3, 10, 100, len(assertion_scores)]
            hit_k_values = []

            true_samples = [inputs for inputs, _ in test_concept_assertions if inputs[0] == concept_idx] # This is problematic when dealing with big datasets

            #print(f'True samples in evaluation dset: {true_samples}')

            for k in k_list:
                hit_k = any(torch.equal(scored_sample[0], true_sample) for true_sample in true_samples for scored_sample in sorted_scores[:k])
                hit_k_values.append(hit_k)
                #print(f'Top{k}hits: {hit_k}')
            
            hits.append(hit_k_values)

            top1 += int(hit_k_values[0])
            top3 += int(hit_k_values[1])
            top10 += int(hit_k_values[2])
            top100 += int(hit_k_values[3])
            top_all += int(hit_k_values[4])

    hits_at_k = [round(sum(hit_values) / len(hit_values), 3) for hit_values in zip(*hits)]  # Calculate hits_at_k for each k

    # return hits_at_k, [top1, top3, top10, top100, top_all]
    return hits_at_k


'''
Computes hits@k for role assertions.
'''

def get_hits_at_k_role_assertions(model, GeoInterp_dataclass,
                  test_concept_assertions, test_role_assertions,
                  entity_to_idx_vocab: dict, idx_to_entity_vocab: dict,
                  idx_to_concept_vocab: dict, idx_to_role_vocab: dict,
                  centroid_score = False
                  ):
    
    top1 = 0
    top3 = 0
    top10 = 0
    top100 = 0
    top_all = 0

    model.eval()

    hits = []
    relevant_assertions = []

    # Convert PyTorch dataset to a numpy array for vectorization
    assertions_array = [assertion[0].numpy() for assertion in test_role_assertions]
    assertions_array = np.stack(assertions_array)

    '''
    The array below is used to disregard duplicate queries.
    For ex., if we have two assertions r(a,b) and r(a,c), the function
    will treat r(a, ?) as a query with b and c as positive answers. It
    will then disregard any other.
    '''

    filter_array = np.ones((assertions_array.shape), dtype=int)
    filter_counter = 0

    with torch.no_grad():

        for assertion_idx, assertion in enumerate(assertions_array):

            filter_counter = assertion_idx

            if np.all(filter_array[filter_counter] == 1):

                head_entity_idx = assertion[0]
                role_entity_idx = assertion[1]
                filter_arr = (assertions_array[:, 0] == head_entity_idx) & (assertions_array[:, 1] == role_entity_idx)
                relevant_assertions_idcs = np.where(filter_arr)[0]
                relevant_assertions = torch.tensor(np.array([assertions_array[idx] for idx in relevant_assertions_idcs]))
                filter_array[relevant_assertions_idcs] = 0

                assertion_scores = []

                for _, tail_entity_idx in entity_to_idx_vocab.items():
                    eval_sample = torch.tensor([head_entity_idx, role_entity_idx, tail_entity_idx]).unsqueeze(0)
                    outputs1, outputs2, outputs3 = model(eval_sample)

                    if centroid_score == False:
                        assertion_score = torch.dist(outputs1, outputs2, p=2)                                                                           
                    else:
                        assertion_score = torch.dist(outputs1, outputs2, p=2) + torch.dist(outputs2, torch.tensor(GeoInterp_dataclass.role_geointerps_dict[idx_to_role_vocab[role_entity_idx]].centroid)) # Change this random call to the GeoInterp class without it being passed

                    assertion_scores.append((torch.tensor([head_entity_idx, role_entity_idx, tail_entity_idx]), assertion_score.item()))

                sorted_scores = sorted(assertion_scores, key=lambda x: x[1])

                k_list = [1, 3, 10, 100, len(assertion_scores)]
                hit_k_values = []

                for k in k_list:
                    hit_k = any(torch.equal(scored_sample[0], assertion) for assertion in relevant_assertions for scored_sample in sorted_scores[:k])
                    hit_k_values.append(hit_k)
            
                hits.append(hit_k_values)

                top1 += int(hit_k_values[0])
                top3 += int(hit_k_values[1])
                top10 += int(hit_k_values[2])
                top100 += int(hit_k_values[3])
                top_all += int(hit_k_values[4])

            else:
                pass

    hits_at_k = [round(sum(hit_values) / len(hit_values), 3) for hit_values in zip(*hits)]  # Calculate hits_at_k for each k
    # print(f'Hits at 1, 3, 10, 100 and all: {hits_at_k}')

    return hits_at_k

'''
Helper function to plot the loss after training a model.
'''

def plot_loss(train_loss, test_loss, num_epoch):
    
    plt.plot(range(1, num_epoch+1), train_loss, 'b-', label='Train Loss')
    plt.plot(range(1, num_epoch+1), test_loss, 'r-', label='Test Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss per Epoch')
    plt.legend()

    # Display the plot
    plt.show()


'''
Helper function to save models hparams and scores as a dictionary.
'''
def save_model(centroid_score, lr,
               phi, gamma, emb_dim, epochs,
               log_epoch, eval_freq,
               eval_test,
               loss_fn, model, optimizer,
               train_loss_list, test_loss_list,
               train_hits_at_k_concept, test_hits_at_k_concept,
               train_hits_at_k_role, test_hits_at_k_role):
    
    model_hparams = {'centroid_score': centroid_score,
                     'lr': lr,
                     'phi': phi,
                     'gamma': gamma,
                     'emb_dim': emb_dim,
                     'epochs': epochs,
                     'log_epoch': log_epoch,
                     'eval_freq': eval_freq,
                     'eval_test': eval_test,
                     'loss_fn': loss_fn,
                     'model': model,
                     'optimizer': optimizer,
                     'train_loss_list': train_loss_list,
                     'test_loss_list': test_loss_list,
                     'train_hits_at_k_concept': train_hits_at_k_concept,
                     'test_hits_at_k_concept': test_hits_at_k_concept,
                     'train_hits_at_k_role': train_hits_at_k_role,
                     'test_hits_at_k_role': test_hits_at_k_role,
                     'misc notes': []
                     }
    
    return model_hparams

def plot_score_hak(hits_at_k_concept, hits_at_k_roles, topk, num_epoch, eval_freq):

    concept_hits_at_topk = [score_list[topk] for score_list in hits_at_k_concept]
    roles_hits_at_topk = [scores[topk] for scores in hits_at_k_roles]

    hak_dict = {0: 1,
                1: 3,
                2: 10,
                3: 100,
                4: 'all'}
    
    plt.plot(range(1, num_epoch+1, eval_freq), concept_hits_at_topk, 'b-', label=f'H@{hak_dict[topk]} concepts')

    try:
        plt.plot(range(1, num_epoch+1, eval_freq), roles_hits_at_topk, 'r-', label=f'H@{hak_dict[topk]} roles')
    except:
        print('No roles to plot.')

    plt.ylim(0, 1.02)
    plt.xlabel('Epochs')
    plt.ylabel(f'hits@{hak_dict[topk]}')
    plt.title(f'Hits@{hak_dict[topk]} every {eval_freq} epochs')
    plt.legend()

    plt.show()

'''
Helper function for representing two dimensions of the models params graphically.
'''

def plot_model(model, GeoInterp_dataclass, individual_vocab_idcs, concept_vocab_idcs, role_vocab_idcs, scaling_factor, dim1, dim2):

    individual_embeddings = model.individual_embedding_dict.weight
    concept_parameter_embeddings = model.concept_embedding_dict.weight
    role_parameter_embeddings = model.role_embedding_dict.weight

    individuals_for_plotting = []
    concept_parameters_for_plotting = []
    concept_centroid_for_plotting = []
    role_parameters_for_plotting = []
    role_centroid_for_plotting = []

    for idx, individual in enumerate(individual_embeddings[:]):
        individual = individual[:].detach().numpy()
        individual_label = individual_vocab_idcs[idx]
        final_representation = (individual, individual_label)
        individuals_for_plotting.append(final_representation)

    for idx, concept in enumerate(concept_parameter_embeddings):
        concept_param = concept[:].detach().numpy()
        concept_label = concept_vocab_idcs[idx]
        final_representation = (concept_param, concept_label)
        concept_parameters_for_plotting.append(final_representation)

    for idx, key in enumerate(GeoInterp_dataclass.concept_geointerps_dict.keys()):
        concept_centroid = GeoInterp_dataclass.concept_geointerps_dict[key].centroid[:]
        concept_label = key + '_centroid'
        final_representation = (concept_centroid, concept_label)
        concept_centroid_for_plotting.append(final_representation)

    for idx, role in enumerate(role_parameter_embeddings):
        role_param = role[:].detach().numpy()
        role_label = role_vocab_idcs[idx]
        final_representation = (role_param, role_label)
        role_parameters_for_plotting.append(final_representation)

    for idx, key in enumerate(GeoInterp_dataclass.role_geointerps_dict.keys()):
        role_centroid = GeoInterp_dataclass.role_geointerps_dict[key].centroid[:]
        role_label = key + '_centroid'
        final_representation = (role_centroid, role_label)
        role_centroid_for_plotting.append(final_representation)


    # Create a figure and axis object
    fig, ax = plt.subplots()

    ax.set_xlim(-1, scaling_factor + scaling_factor/10)
    ax.set_ylim(-1, scaling_factor + scaling_factor/10)
    ax.grid(True)

    ax.plot(0, 0, 'yo')

    # Plot individual points in blue
    for individual, label in individuals_for_plotting:
        ax.plot(individual[dim1], individual[dim2], 'bo', label=label)
        ax.annotate(label, xy=(individual[dim1], individual[dim2]), xytext=(3, -3), textcoords='offset points')

    # Plot concept points in red
    for concept_param, label in concept_parameters_for_plotting:
        ax.plot(concept_param[dim1], concept_param[dim2], 'r+', label=label)
        ax.annotate(label, xy=(concept_param[dim1], concept_param[dim2]), xytext=(3, -3), textcoords='offset points')

    for concept_centroid, label in concept_centroid_for_plotting:
        ax.plot(concept_centroid[dim1], concept_centroid[dim2], 'go', label=label)
        ax.annotate(label, xy=(concept_centroid[dim1], concept_centroid[dim2]), xytext=(3, -3), textcoords='offset points')

    # Plot role points in yellow
    
    for role_param, label in role_parameters_for_plotting:
        ax.plot(role_param[dim1], role_param[dim2], 'y+', label=label)
        ax.annotate(label, xy=(role_param[dim1], role_param[dim2]), xytext=(3, -3), textcoords='offset points')
        
    for role_centroid, label in role_centroid_for_plotting:
        ax.plot(role_centroid[dim1], role_centroid[dim2], 'yo', label=label)
        ax.annotate(label, xy=(role_centroid[dim1], role_centroid[dim2]), xytext=(3, -3), textcoords='offset points')

    plt.show()


''' 
Helper function for plotting H@K evals per increment.
'''

def plot_score_hak(hits_at_k_concept, hits_at_k_roles, topk, num_epoch, eval_freq):

    concept_hits_at_topk = [score_list[topk] for score_list in hits_at_k_concept]
    roles_hits_at_topk = [scores[topk] for scores in hits_at_k_roles]

    hak_dict = {0: 1,
                1: 3,
                2: 10,
                3: 100,
                4: 'all'}
    
    plt.plot(range(1, num_epoch+1, eval_freq), concept_hits_at_topk, 'b-', label=f'H@{hak_dict[topk]} concepts')

    try:
        plt.plot(range(1, num_epoch+1, eval_freq), roles_hits_at_topk, 'r-', label=f'H@{hak_dict[topk]} roles')
    except:
        print('No roles to plot.')

    plt.ylim(0, 1.02)
    plt.xlabel('Epochs')
    plt.ylabel(f'hits@{hak_dict[topk]}')
    plt.title(f'Hits@{hak_dict[topk]} every {eval_freq} epochs')
    plt.legend()

    plt.show()


'''
Main training loop.
'''

def train(model, concept_dataloader, role_dataloader, loss_fn, optimizer, neg_sampling = bool):
    model.train()
    total_loss = 0.0
    num_batches = len(concept_dataloader)

    for i, data in enumerate(role_dataloader):
        model.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs1, outputs2, outputs3 = model(inputs) # outputs1 = Role Parameter, outputs2 = EntitySubj concat parameter, outputs3 = neg_candidate

        if neg_sampling == True:
            loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels)) + -torch.dist(outputs2, outputs3, p=2)
        else:
            loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels))

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        model.role_parameter_constraint()

    for i, data in enumerate(concept_dataloader):
        model.train()
        inputs, labels = data
        optimizer.zero_grad()
        outputs1, outputs2, outputs3 = model(inputs) # Outputs 1 = Concept Parameter, Outputs 2 = Entity Parameter, Outputs 3 = neg_candidate

        if neg_sampling == True:
            loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels)) + -torch.dist(outputs2, outputs3, p=2)
        else:
            loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels))
            
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        model.concept_parameter_constraint()

    return total_loss / num_batches

''' 
Function for obtaining test loss.
'''

def test(model, concept_dataloader, role_dataloader, loss_fn, neg_sampling = bool):
    model.eval()
    total_loss = 0.0
    num_batches = len(concept_dataloader)

    with torch.no_grad():

        for i, data in enumerate(role_dataloader):
            inputs, labels = data
            outputs1, outputs2, outputs3 = model(inputs) # outputs1 = Role Parameter, outputs2 = Entity concat parameter, outputs3 = Entity concat neg_candidate
            if neg_sampling == True:
                loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels)) + -torch.dist(outputs2, outputs3, p=2)
            else:
                loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels))

            total_loss += loss.item()

        for i, data in enumerate(concept_dataloader):
            inputs, labels = data
            outputs1, outputs2, outputs3 = model(inputs) # Outputs 1 = Concept Parameter, Outputs 2 = Entity Parameter, Outputs 3 = Entity concat neg_candidate
            if neg_sampling == True:
                loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels)) + -torch.dist(outputs2, outputs3, p=2)
            else:
                loss = (loss_fn(outputs2, labels) + model.gamma * loss_fn(outputs1, outputs2) + model.phi * loss_fn(outputs1, labels))
                
            total_loss += loss.item()

    return total_loss / num_batches

'''
Main function for training.
'''

def train_model(model, GeoInterp_dataclass,
                train_concept_loader, train_role_loader,
                test_concept_loader, test_role_loader,
                train_concept_dset, test_concept_dset,
                train_role_dset, test_role_dset,
                num_epochs, loss_log_freq,
                eval_freq, eval_train,
                loss_function, optimizer,
                idx_to_entity: dict, entity_to_idx: dict,
                idx_to_concept: dict, concept_to_idx: dict,
                idx_to_role: dict, role_to_idx: dict,
                centroid_score = False, neg_sampling = False,
                plot_loss_flag = False
                ):

    train_loss_list = []
    test_loss_list = []

    train_hits_at_k_concept = []
    test_hits_at_k_concept = []

    train_hits_at_k_role = []
    test_hits_at_k_role = []


    for epoch in range(1, num_epochs + 1):
        train_loss = train(model, train_concept_loader, train_role_loader, loss_function, optimizer, neg_sampling)
        train_loss_list.append(train_loss)
        test_loss = test(model, test_concept_loader, test_role_loader, loss_function, neg_sampling)
        test_loss_list.append(test_loss)

        if epoch % loss_log_freq == 0:
            print(f'Epoch {epoch}/{num_epochs} -> Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}\n')

        if epoch % eval_freq == 0:
            print(f'Epoch {epoch}: Initiating evaluation. \n')
            
            try:
                test_hak_concept = get_hits_at_k_concept_assertions(model, GeoInterp_dataclass, test_concept_dset, test_role_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                test_hits_at_k_concept.append(test_hak_concept)
            except:
                print('Exception found. H@K for the Concept Test Dataset have not been computed.')
                pass

            if eval_train == True:
                try:
                    train_hak_concept = get_hits_at_k_concept_assertions(model, GeoInterp_dataclass, train_concept_dset, train_role_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                    train_hits_at_k_concept.append(train_hak_concept)
                except:
                    print('Exception found. H@K for the Concept Train Dataset have not been computed.')
                    pass
            
            try:
                test_hak_role = get_hits_at_k_role_assertions(model, GeoInterp_dataclass, test_concept_dset, test_role_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                test_hits_at_k_role.append(test_hak_role)
            except:
                print('Exception found. H@K for the Role Test Dataset have not been computed.')
                pass
            
            if eval_train == True:
                try:
                    train_hak_role = get_hits_at_k_role_assertions(model, GeoInterp_dataclass, train_concept_dset, train_role_dset, entity_to_idx, idx_to_entity, idx_to_concept, idx_to_role, centroid_score)
                    train_hits_at_k_role.append(train_hak_role)
                except:
                    print('Exception found. H@K for the Role Train Dataset have not been computed.')
                    pass
    
    if plot_loss_flag == True:
        plot_loss(train_loss_list, test_loss_list, num_epochs)

    return train_loss_list, test_loss_list, train_hits_at_k_concept, test_hits_at_k_concept, train_hits_at_k_role, test_hits_at_k_role