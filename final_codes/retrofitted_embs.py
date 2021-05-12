import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tqdm
from tqdm import tqdm
from scipy.spatial import distance
import sys
import pandas as pd
import matplotlib.pyplot as plt
from random import sample
import statistics as st

from retrofitting_funcs import get_embs, mean_centering, format_vector, top_d_var, dim_removal, retrofit
from sim_funcs import get_self_sim, get_avg_sense_sim, get_inter_sim, get_wf, get_sf, sim_combine

model_type = sys.argv[1]


os.chdir('../WSD_data/mydata/')

datasets = ['semeval15_13', 'senseval3task1', 'senseval2', 'semeval13_12', 'semeval07_7']


data = {}
van = {}
words = {}
van_vec = {}

for layer in range(13):
    for data_num in datasets:
        print('doing for ', str(data_num))
        data[data_num] = get_embs(data_num, layer, model_type)
        van[data_num] = [((y[0][0], y[0][1], y[0][2]), y[1]) for x in data[data_num] for y in x]
        words[data_num] = list(set([x[0][1] for x in van[data_num]]))
    
        #with centering (dim and dim+retro)
        van_cent = format_vector(mean_centering(van[data_num]), van[data_num])
        van_cent_dim = format_vector(dim_removal(van_cent, 1), van[data_num])
        van_cent_dim_retro = format_vector(retrofit(van_cent_dim), van[data_num])
    
        #saving files
        os.chdir('../emb_data/'+str(model_type)+'/layer_'+str(layer)+'/')
            
        #saving LA vectors
        with open(str(data_num)+'_1_cent_embs.pkl', 'wb') as f1:
            pickle.dump(van_cent_dim, f1)
            
        #saving LASeR vectors
        with open(str(data_num)+'_1_retro_embs.pkl', 'wb') as f1:
            pickle.dump(van_cent_dim_retro, f1)
        

print('done!')
