from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import linalg as LA
import random
from tqdm import tqdm
import torch 

def self_similarity_and_rogue(all_hidden_states, token_list, inference_list, n_layers_start = 0, n_layers = 13, analyze_rogue = False):
    all_layer_self_sim = {} # store all self-similarity data for all layers

    self_sim_rogue_dimensions = {}

    for layer in tqdm(range(n_layers_start, n_layers), desc='Layer Progress'):
        layer_data = {} # store self-similarity data for just current layer
        rogue = {}

        temp = all_hidden_states[layer]
        
        for token in tqdm(token_list, desc='Token Progress'):
            token_embeddings = [] # a list to store all embeddings of that word in different contexts

            for (sentence_index, token_index) in inference_list[token]:
            
                token_embeddings.append(np.array(temp[sentence_index][token_index]))
            token_embeddings = np.array(token_embeddings)

            sim_matrix = cosine_similarity(token_embeddings, token_embeddings)

            if analyze_rogue == True:
                cp = identifying_rogue_dimensions(token_embeddings)
            else:
                cp = []

            if len(sim_matrix) != 1:

                self_similarity = round((np.sum(sim_matrix) - len(sim_matrix))/(len(sim_matrix) * (len(sim_matrix)-1)),3)
                layer_data[token] = self_similarity
                rogue[token] = cp

        all_layer_self_sim[layer] = layer_data
        self_sim_rogue_dimensions[layer] = rogue

    return all_layer_self_sim, self_sim_rogue_dimensions

def identifying_rogue_dimensions(matrix): # matrix refers to the embedding matrix

    embedding_norm = LA.norm(matrix, axis = 1)
    embedding_norm = np.expand_dims(embedding_norm, axis=0)
    
    cos= cosine_similarity(matrix, matrix)
    total = np.sum(cos)
    
    contributions = []

    for n in range(matrix.shape[1]):  # matrix.shape[1] means the number of dimensions of the embeddings
        dc = np.dot(matrix[:,n:n+1], matrix[:,n:n+1].T) # dimension contribution #getting u_i and v_i
        dc = dc/np.dot(embedding_norm.T, embedding_norm)
        contributions.append(np.sum(dc)/total)

    return contributions


def mean_pooling(all_hidden_states, attention_mask, layer):
    token_embeddings = all_hidden_states[layer]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def intra_sim_and_norm(all_hidden_states, sentence_list, encoded_input, n_layers_start = 0, n_layers = 13):

    all_layer_intra_sim = {}

    all_layer_norm = {}

    token_length = encoded_input['attention_mask'].sum(axis = 1)## this computes token length, ignoring padding token for computing intra-sim 

    for layer in tqdm(range(n_layers_start, n_layers), desc='layer progress'):

        sentence_embeddings = mean_pooling(all_hidden_states, encoded_input['attention_mask'], layer)
        temp = all_hidden_states[layer]

        layer_intra_sim = []
        layer_norm = []

        for sent in tqdm(range(len(sentence_list)), desc = 'sentence progress'):
        
            word_in_sent_embeddings = temp[sent][:token_length[sent]] ## indexed sentence, indexed tokens that that are attended

            intra_sim = cosine_similarity(word_in_sent_embeddings, sentence_embeddings[sent][None,:]).mean()

            layer_intra_sim.append(intra_sim)
            layer_norm.extend([LA.norm(emb) for emb in word_in_sent_embeddings])

        all_layer_intra_sim[layer] = layer_intra_sim

        all_layer_norm[layer] = layer_norm
        
    return all_layer_intra_sim, all_layer_norm


def anisotropy_baseline(sentence_list, all_hidden_states, encoded_input, n_samples = 1000, n_layers = 13):

    sentence_indices = random.sample(list(range(len(sentence_list))), n_samples) # to select from n sentences to perform analysis on

    token_length = encoded_input['attention_mask'].sum(axis = 1)## this computes token length, ignoring padding token for computing intra-sim 
    
    all_layer_anisotropy = {}

    for layer in tqdm(range(n_layers), desc='layer progress'):

        layer_sampled_word_embeddings = []

        temp = all_hidden_states[layer]

        for sent in sentence_indices:
            
            # non-padding word embeddings from that sentence
            word_in_sent_embeddings = temp[sent][:token_length[sent]] ## indexed sentence, indexed tokens that that are attended
            
            # randomly sample just one word from the sentence
            # to avoid the sampling biased towards longer sentences
            layer_sampled_word_embeddings.append(word_in_sent_embeddings[random.choice(list(range(len(word_in_sent_embeddings))))].tolist())
        
        anisotropy_matrix = cosine_similarity(np.array(layer_sampled_word_embeddings), np.array(layer_sampled_word_embeddings))

        anisotropy = round((np.sum(anisotropy_matrix) - len(anisotropy_matrix))/(len(anisotropy_matrix) * (len(anisotropy_matrix)-1)),3) # round to three decimal cuz lost of precision

        all_layer_anisotropy[layer] = anisotropy

    return anisotropy_matrix, all_layer_anisotropy