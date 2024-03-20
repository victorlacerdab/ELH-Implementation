import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, random_split
import torch
from main import GeometricInterpretation

def plot_loss(train_loss, test_loss, num_epoch):
    
    plt.plot(range(1, num_epoch+1), train_loss, 'b-', label='Train Loss')
    plt.plot(range(1, num_epoch+1), test_loss, 'r-', label='Test Loss')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss per Epoch')
    plt.legend()

    # Display the plot
    plt.show()


def get_hits_at_k_concept_assertions(model,
                  test_concept_assertions=Dataset, test_role_assertions=Dataset,
                  entity_to_idx_vocab=dict, idx_to_entity_vocab=dict,
                  idx_to_concept_vocab=dict, idx_to_role_vocab=dict,
                  centroid_score = False
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

    with torch.no_grad():

        for concept_idx in relevant_concept_idx:
            assertion_scores = []

            for _, entity_idx in entity_to_idx_vocab.items():
                eval_sample = torch.tensor([concept_idx, entity_idx]).unsqueeze(0)
                outputs1, outputs2 = model(eval_sample) # out1 = Concept parameter, out2 = Individual parameter

                if centroid_score == False:
                    assertion_score = torch.dist(outputs1, outputs2, p=2)
                else:
                    assertion_score = torch.dist(outputs1, outputs2, p=2) + torch.dist(outputs2, torch.tensor(GeometricInterpretation.concept_geointerps_dict[idx_to_concept_vocab[int(concept_idx)]].centroid))

                assertion_scores.append((torch.tensor([concept_idx, entity_idx]), assertion_score.item()))
            
            sorted_scores = sorted(assertion_scores, key=lambda x: x[1])

            k_list = [1, 3, 10, 100, len(assertion_scores)]
            hit_k_values = []

            true_samples = [inputs for inputs, _ in test_concept_assertions if inputs[0] == concept_idx] # This is problematic when dealing with big datasets

            for k in k_list:
                hit_k = any(torch.equal(scored_sample[0], true_sample) for true_sample in true_samples for scored_sample in sorted_scores[:k])
                hit_k_values.append(hit_k)
            
            hits.append(hit_k_values)

            top1 += int(hit_k_values[0])
            top3 += int(hit_k_values[1])
            top10 += int(hit_k_values[2])
            top100 += int(hit_k_values[3])
            top_all += int(hit_k_values[4])

    hits_at_k = [round(sum(hit_values) / len(hit_values), 3) for hit_values in zip(*hits)]  # Calculate hits_at_k for each k

    # return hits_at_k, [top1, top3, top10, top100, top_all]
    return hits_at_k



def get_hits_at_k_role_assertions(model,
                  test_concept_assertions=Dataset, test_role_assertions=Dataset,
                  entity_to_idx_vocab=dict, idx_to_entity_vocab=dict,
                  idx_to_concept_vocab=dict, idx_to_role_vocab=dict,
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
                    outputs1, outputs2 = model(eval_sample)
                    if centroid_score == False:
                        assertion_score = torch.dist(outputs1, outputs2, p=2)
                    else:
                        assertion_score = torch.dist(outputs1, outputs2, p=2) + torch.dist(outputs2, torch.tensor(GeometricInterpretation.role_geointerps_dict[idx_to_role_vocab[role_entity_idx]].centroid))

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

    # return hits_at_k, [top1, top3, top10, top100, top_all]
    return hits_at_k

def get_param_dists(model, dictionary_concept_to_idx, role = False):

    dist_dict = {}

    for predicate, idx in dictionary_concept_to_idx.items():
        with torch.no_grad():
            if role == False:
                dist = torch.dist(torch.tensor(GeometricInterpretation.concept_geointerps_dict[predicate].centroid), model.concept_embedding_dict.weight[idx])
                dist_dict[predicate] = dist
            else:
                dist = torch.dist(torch.tensor(GeometricInterpretation.role_geointerps_dict[predicate].centroid), model.concept_embedding_dict.weight[idx])
                dist_dict[predicate] = dist

    return dist_dict

def plot_score_hak(hits_at_k_concept, hits_at_k_roles, topk, num_epoch, eval_freq):

    concept_hits_at_topk = [score_list[topk] for score_list in hits_at_k_concept]
    roles_hits_at_topk = [scores[topk] for scores in hits_at_k_roles]

    hak_dict = {0: 1,
                1: 3,
                2: 10,
                3: 100,
                4: 'all'}
    
    plt.plot(range(1, num_epoch+1, eval_freq), concept_hits_at_topk, 'b-', label=f'H@{hak_dict[topk]} concepts')

    plt.plot(range(1, num_epoch+1, eval_freq), roles_hits_at_topk, 'r-', label=f'H@{hak_dict[topk]} roles')

    plt.ylim(0, 1.02)
    plt.xlabel('Epochs')
    plt.ylabel(f'hits@{hak_dict[topk]}')
    plt.title(f'Hits@{hak_dict[topk]} every {eval_freq} epochs')
    plt.legend()

    plt.show()