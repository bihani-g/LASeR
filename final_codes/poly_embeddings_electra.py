##polysemous embedding extraction
import os
import sys
import bs4
from bs4 import BeautifulSoup
import pickle
import nltk
import numpy as np
from nltk.corpus import semcor
import torch
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, XLNetTokenizer, XLNetModel, ElectraConfig, ElectraModel, ElectraTokenizer
import pandas as pd
from scipy.spatial import distance
import h5py
import tqdm 
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle
# from functions import map_token_senses, get_hidden_states

ds_type = sys.argv[1]
model_type = 'electra'

os.chdir('/media/disk4/context_div/WSD_data/mydata/')

print('getting senses', flush = True)
with open(str(ds_type)+'_senses.pkl', 'rb') as f:
    senses = pickle.load(f)

print('getting sentences', flush = True)
with open(str(ds_type)+'_sents.pkl', 'rb') as f:
    sents = pickle.load(f)

print('getting sentence ids for sentences with polysemous words', flush = True)
with open(str(ds_type)+'_poly.pkl', 'rb') as f:
    poly = pickle.load(f)

    
#sents = sents[0:1000]
#senses = [x for x in senses if x[0]<1000]


if model_type == 'bert':
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
elif model_type == 'gpt2':
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", do_lower_case=True)
    model = GPT2Model.from_pretrained('gpt2',output_hidden_states = True)
elif model_type == 'xlnet':
    tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased', do_lower_case=True, remove_space =True, 
                                           add_special_tokens = False)
    model = XLNetModel.from_pretrained('xlnet-base-cased',output_hidden_states = True)
elif model_type == 'electra':
    config = ElectraConfig.from_pretrained('google/electra-small-discriminator', hidden_size = 768)
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator', do_lower_case=True)
    model = ElectraModel(config)

model.eval()


print('extracting hidden states and mapping with word and sense values', flush = True)

for layer in range(13):
    os.chdir('/media/disk4/context_div/emb_data/'+str(model_type)+'/layer_'+str(layer)+'/')
#     os.chdir('/media/disk4/context_div/test/')
    for j in tqdm(range(len(sents))):
        with open(str(ds_type)+'_embs.pkl', 'ab') as f:
            poly_words = [x[1] for x in poly if x[0] == j]
            if len(poly_words) == 1:
                poly_words = poly_words[0]
                toks = []
    
                for i in sents[j][1].split(" "):
                    inp_ids = tokenizer(i, add_special_tokens=False, return_tensors="pt").input_ids[0]
                    toks.append(tokenizer.convert_ids_to_tokens(inp_ids))
        
                    #get senses for this sentence
                senses_i = [(x[1], x[2]) for x in senses if x[0] == j]
                sense_toks_i = list(zip(senses_i, toks))
                sense_toks_flat = [(x[0][0], x[0][1]) for x in sense_toks_i for y in x[1]]
                poly_toks_flat = []
                for ch in range(len(sense_toks_flat)):
                    if sense_toks_flat[ch][0] in poly_words:
                        poly_toks_flat.append((ch, sense_toks_flat[ch][0], sense_toks_flat[ch][1]))
            
                hidden_states = []
        
                batch = tokenizer(sents[j][1], add_special_tokens=False, return_tensors="pt")
                input_ids = batch.input_ids
#                 attention_mask = batch.attention_mask
    
    
                with torch.no_grad():
                    last_hidden_states = model(input_ids, output_hidden_states = True)
#                 print(last_hidden_states[1][layer][0].shape)
                    hidden_states.append(last_hidden_states[1][layer][0])   
    
                hidden_states = [y for x in hidden_states for y in x]
                poly_hidden= []
                for hh in range(len(hidden_states)):
                    if hh in [x[0] for x in poly_toks_flat]:
                        poly_hidden.append(hidden_states[hh])
    
    
                sense_embeds = list(zip(poly_toks_flat, poly_hidden))
#             print(sense_embeds[0][1])


                pickle.dump(sense_embeds, f)


            else:
                continue
                
print('done!', flush = True)