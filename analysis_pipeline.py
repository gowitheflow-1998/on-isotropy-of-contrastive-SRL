import argparse
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
import itertools
from collections import Counter
import torch
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy import linalg as LA
import random
from tqdm import tqdm
from metrics import self_similarity_and_rogue, identifying_rogue_dimensions, mean_pooling, intra_sim_and_norm, anisotropy_baseline

class MyDataset(Dataset):
    def __init__(self, encoded_inputs):
        self.encoded_inputs = encoded_inputs

    def __len__(self):
        return self.encoded_inputs["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encoded_inputs.items()}

class AnalysisPipeline:

    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,
                                        output_hidden_states = True # returns all hidden-states, enabling analysis on all layers
                                        )
        print("Initializing Analysis Pipeline with %s"%model_name)

    def data(self, dataset_name, dataset_column):
        if dataset_name == "stsb_multi_mt":
            dataset = load_dataset("stsb_multi_mt", name="en", split="test")
        else:
            try:
                dataset = load_dataset(dataset_name)
            except ValueError:
                print(f"Error: Dataset '{dataset_name}' is not available.")

        sentences = dataset[dataset_column]
        return sentences
    
    def tokenize_and_preprocessing(self, sentences):
        print("Tokenizing text with model:", self.model_name)
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')# max length should be added here instead of when initializing it    
        print("Finished Tokenizing:", self.model_name)
        # print(encoded_input['input_ids'][0])
        tokenized = encoded_input['input_ids'].tolist()
        token_list = list(itertools.chain.from_iterable(tokenized))
        counter = Counter(token_list)

        # get the words to analyze self similarity - appearing in more than 5 contexts (some contexts) at least
        index_count = [(index, count) for (index,count) in counter.most_common() if count >=5]
        token_list = [index for (index,count) in counter.most_common() if count >=5]
        return encoded_input, tokenized, index_count, token_list
    
    def encode(self, encoded_input, batch_size = 32):
        dataset = MyDataset(encoded_input)
        dataloader = DataLoader(dataset, batch_size=batch_size)  # Adjust batch_size as needed
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = self.model.to(device)
        print("Encoding text")
        # Placeholders for the model outputs
        total_output = []
        total_hidden_states = []

        # Iterate over the dataloader
        with torch.no_grad():  # Disable gradient calculation
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}  # Move the batch tensor to the right device
                model_output = model(**batch)
                total_output.append(model_output.last_hidden_state.detach().cpu())
                total_hidden_states.append(tuple(state.detach().cpu() for state in model_output.hidden_states))

        # Concatenate all outputs and hidden states
        total_output = torch.cat(total_output, dim=0)
        total_hidden_states = tuple(torch.cat(state, dim=0) for state in zip(*total_hidden_states))
        print("Finished encoding")
        return total_output, total_hidden_states

    def position_list(self, set_of_token, list_of_tokenized_sentences):

        inference_list = {} ## get a dict to store corresponding position (sentence_index, token_index) of words
        for n in set_of_token:
            position_list = []
            for sen_index, sen in enumerate(list_of_tokenized_sentences):
                if n in sen:
                    token_index = sen.index(n)
                    position_list.append((sen_index, token_index)) # store the corresponding sentence index and token index of that word
            inference_list[n] = position_list

        return inference_list    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--model_name', type=str, default="sentence-transformers/all-mpnet-base-v2", help='model name or path')
    parser.add_argument('--dataset_name', type=str, default="stsb_multi_mt", help="dataset name")
    parser.add_argument('--dataset_column', type=str, default="sentence1", help="dataset column to analyze")
    parser.add_argument('--encoding_batch_size', type=int, default=64, help="encoding batch size")
    parser.add_argument('--analyze_layer_start', type=int, default=0, help="encoding batch size")
    parser.add_argument('--analyze_layer_end', type=int, default=None, help="encoding batch size")

    args = parser.parse_args()
    
    analysis = AnalysisPipeline(args.model_name)
    num_layers = analysis.model.config.num_hidden_layers
    sentences = analysis.data(args.dataset_name, args.dataset_column)
    encoded_input, tokenized, index_count, token_list = analysis.tokenize_and_preprocessing(sentences)
    _, total_hidden_states = analysis.encode(encoded_input, batch_size = 64)
    inference_list = analysis.position_list(token_list, tokenized)

    if args.analyze_layer_start < 0 or args.analyze_layer_start > num_layers:
        raise ValueError("Invalid value for 'analyze_layer_start'. It cannot be less than 0 or greater than number of layers of the model")
    else:
        n_layers_start = args.analyze_layer_start

    n_layers_start = args.analyze_layer_start
    if args.analyze_layer_end != None:
        if args.analyze_layer_end <= num_layers+1:
            n_layers = args.analyze_layer_end
        else:
            raise ValueError("Invalid value for 'analyze_layer_end'. It cannot be greater than number of layers of the model + 1.")
    else:
        n_layers = num_layers+1

    print("analyzing starting from layer %s till layer %s"%(str(n_layers_start), str(n_layers-1)))

    all_layer_self_sim, self_sim_rogue_dimensions = self_similarity_and_rogue(total_hidden_states, 
                                                                              token_list, 
                                                                              inference_list, 
                                                                              n_layers_start = n_layers_start, 
                                                                              n_layers = n_layers, 
                                                                              analyze_rogue = True)
    
    all_layer_intra_sim, all_layer_norm = intra_sim_and_norm(total_hidden_states,
                                                              sentences, 
                                                              encoded_input, 
                                                              n_layers_start = n_layers_start, 
                                                              n_layers = n_layers)
    
    anisotropy_matrix, all_layer_anisotropy = anisotropy_baseline(sentences, 
                                                                  total_hidden_states, 
                                                                  encoded_input, 
                                                                  n_samples = 1000, 
                                                                  n_layers = n_layers)

    
    unadjusted_self_sim = np.array([np.mean([value for key,value in all_layer_self_sim[layer].items()]) for layer in range(n_layers_start, n_layers)])
    unadjusted_intra_sim = np.array([np.mean(all_layer_intra_sim[layer]) for layer in range(n_layers_start, n_layers)])
    ani_baseline = np.array([ani for layer, ani in all_layer_anisotropy.items()])
    ani_baseline = ani_baseline[n_layers_start: n_layers]

    adjusted_intra_sim = unadjusted_intra_sim - ani_baseline                                                                                    
    adjusted_self_sim = unadjusted_self_sim - ani_baseline
    with open('analysis_output', 'w') as f:
        f.write('analysis output for %s'%(args.model_name) + '\n')
        for index, layer in enumerate(range(n_layers_start, n_layers)):                                                                                                                                                                 
            raw_self_output = 'unadjusted self_sim of layer %s: %s'%(layer, unadjusted_self_sim[index])
            print(raw_self_output)
            f.write(raw_self_output + '\n')
        for index, layer in enumerate(range(n_layers_start, n_layers)):                                                                                                                                                                 
            raw_intra_output = 'unadjusted intra_sim of layer %s: %s'%(layer, unadjusted_intra_sim[index])
            print(raw_intra_output)
            f.write(raw_self_output + '\n')           
        for index, layer in enumerate(range(n_layers_start, n_layers)):                                                                                                                                                                 
            ani_output = 'anisotropy estimate %s: %s'%(layer, ani_baseline[index])
            print(ani_output)
            f.write(ani_output + '\n')
        for index, layer in enumerate(range(n_layers_start, n_layers)):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            self_output = 'adjusted self_sim of layer %s: %s'%(layer, adjusted_self_sim[index])
            print(self_output)
            f.write(self_output + '\n')
        for index, layer in enumerate(range(n_layers_start, n_layers)):
            intra_output = 'adjusted intra_sim of layer %s: %s'%(layer, adjusted_intra_sim[index])
            print(intra_output)
            f.write(intra_output + '\n')