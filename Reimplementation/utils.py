from torch.utils.data import DataLoader, Dataset, random_split
import torch

def save_model(centroid_score, lr,
               phi, emb_dim, epochs,
               log_epoch, eval_freq,
               eval_test, alt_train,
               entity_centroid_init, concept_centroid_init,
               role_centroid_init, loss_fn, model, optimizer,
               train_loss_list, test_loss_list,
               train_hits_at_k_concept, test_hits_at_k_concept,
               train_hits_at_k_role, test_hits_at_k_role):
    
    model_hparams = {'centroid_score': centroid_score,
                     'lr': lr,
                     'phi': phi,
                     'emb_dim': emb_dim,
                     'epochs': epochs,
                     'log_epoch': log_epoch,
                     'eval_freq': eval_freq,
                     'eval_test': eval_test,
                     'alt_train': alt_train,
                     'entity_centroid_init': entity_centroid_init,
                     'concept_centroid_init': concept_centroid_init,
                     'role_centroid_init': role_centroid_init,
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

def plot_model(model, individual_vocab_idcs, concept_vocab_idcs, scaling_factor):

    individual_embeddings = model.individual_embedding_dict.weight
    concept_parameter_embeddings = model.concept_embedding_dict.weight

    individuals_for_plotting = []
    concept_parameters_for_plotting = []
    centroid_for_plotting = []

    for idx, individual in enumerate(individual_embeddings[:]):
        individual = individual[1:3].detach().numpy()
        individual_label = individual_vocab_idcs[idx]
        final_representation = (individual, individual_label)
        individuals_for_plotting.append(final_representation)

    for idx, concept in enumerate(concept_parameter_embeddings):
        concept_param = concept[1:3].detach().numpy()
        concept_label = concept_vocab_idcs[idx]
        final_representation = (concept_param, concept_label)
        concept_parameters_for_plotting.append(final_representation)

    for idx, key in enumerate(GeometricInterpretation.concept_geointerps_dict.keys()):
        concept_centroid = GeometricInterpretation.concept_geointerps_dict[key].centroid[1:3]
        concept_label = key + '_centroid'
        final_representation = (concept_centroid, concept_label)
        centroid_for_plotting.append(final_representation)

    # Create a figure and axis object
    fig, ax = plt.subplots()

    ax.set_xlim(-1, scaling_factor + scaling_factor/10)
    ax.set_ylim(-1, scaling_factor + scaling_factor/10)
    ax.grid(True)

    ax.plot(0, 0, 'yo')

    # Plot individual points in blue
    for individual, label in individuals_for_plotting:
        ax.plot(individual[0], individual[1], 'bo', label=label)
        ax.annotate(label, xy=(individual[0], individual[1]), xytext=(3, -3), textcoords='offset points')

    # Plot concept points in red
    for concept_param, label in concept_parameters_for_plotting:
        ax.plot(concept_param[0], concept_param[1], 'r+', label=label)
        ax.annotate(label, xy=(concept_param[0], concept_param[1]), xytext=(3, -3), textcoords='offset points')

    for concept_centroid, label in centroid_for_plotting:
        ax.plot(concept_centroid[0], concept_centroid[1], 'go', label=label)
        ax.annotate(label, xy=(concept_centroid[0], concept_centroid[1]), xytext=(3, -3), textcoords='offset points')

    plt.show()