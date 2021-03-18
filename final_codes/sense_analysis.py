##pos retro anisotropy

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
# plt.style.use('science')
from random import sample
import matplotlib.pyplot as plt
import pandas as pd
import statistics
from statistics import mean, stdev
import math

from retrofitting_funcs import get_embs, get_retro_embs, mean_centering, format_vector, top_d_var, dim_removal, retrofit
from sim_funcs import get_self_sim, get_avg_sense_sim, get_inter_sim, get_wf, get_sf, sim_combine, get_sense_sim


datasets = ['semeval15_13', 'senseval3task1', 'senseval2', 'semeval13_12', 'semeval07_7']
# , 'raganato_ALL',  'senseval3task6', 'senseval3task6_train']

os.chdir('/media/disk4/context_div/WSD_data/mydata/')

with open('nouns.pkl', 'rb') as f:
    nouns = pickle.load(f)
    
with open('verbs.pkl', 'rb') as f:
    verbs = pickle.load(f)
    
with open('adjectives.pkl', 'rb') as f:
    adjectives = pickle.load(f)
    
# model_type = 'bert'
model_type = sys.argv[1]

data = {}
data_r = {}
van = {}
van_r = {}
words = {}
words_r = {}
van_vec = {}
van_vec_r = {}

layer_stats_vanilla = []
layer_stats_retro = []

layer_sim = {}

layer_sim['bert'] = [0.074, 0.11, 0.161, 0.164, 0.199, 0.27, 0.287, 0.309, 0.368, 0.413, 0.445, 0.495, 0.364]
layer_sim['gpt2'] = [0.611, 0.713, 0.653, 0.615, 0.57, 0.546, 0.523, 0.515, 0.528, 0.554, 0.596, 0.722, 0.977]
layer_sim['xlnet'] = [0.059, 0.04, 0.083, 0.115, 0.188, 0.197, 0.487, 0.535, 0.798, 0.811, 0.857, 0.972, 0.944]
layer_sim['electra'] = [0.078, 0.082, 0.088, 0.096, 0.098, 0.102, 0.106, 0.109, 0.115, 0.118, 0.13, 0.134, 0.141]


for layer in range(13):
    for data_num in datasets:
        data[data_num] = get_embs(data_num, layer, model_type)
        van[data_num] = [((y[0][0], y[0][1], y[0][2]), y[1]) for x in data[data_num] for y in x]
        van_vec[data_num] = [x[1] for x in van[data_num]]

    wfs_ = []
    vans_ = []
    for data_num in datasets:
        vans_i = van[data_num]
        vans_ = vans_ + vans_i
    
    vans_filt_ = [x for x in vans_ if ((x[0][1] in [y1[0] for y1 in nouns]) or (x[0][1] in [y2[0] for y2 in verbs]) or
                                     (x[0][1] in [y3[0] for y3 in adjectives]))]
    
    words = list(set([x[0][1] for x in vans_filt_]))
    
    ###################
    
    for data_num in datasets:
        data_r[data_num] = get_retro_embs(data_num, layer, model_type, 1)
        van_r[data_num] = [((x[0][0], x[0][1], x[0][2]), x[1]) for x in data_r[data_num]]
        van_vec[data_num] = [x[1] for x in van_r[data_num]]

    wfs_r_ = []
    vans_r_ = []
    for data_num in datasets:
        vans_i_r = van_r[data_num]
        vans_r_ = vans_r_ + vans_i_r
    
    vans_filt_r_ = [x for x in vans_r_ if ((x[0][1] in [y1[0] for y1 in nouns]) or (x[0][1] in [y2[0] for y2 in verbs]) or
                                     (x[0][1] in [y3[0] for y3 in adjectives]))]
    
    words_r = list(set([x[0][1] for x in vans_filt_r_]))
    
    ##################
    van_sims = sim_combine(words, vans_filt_)
    van_r_sims = sim_combine(words_r, vans_filt_r_)
    
    

    ##vanilla

    self_s = pd.Series([x[4]-layer_sim[model_type][layer] for x in van_sims if x[1]>x[2] and x[5] != 1])
    sense_s = pd.Series([x[5]-layer_sim[model_type][layer] for x in van_sims if x[1]>x[2] and x[5] != 1])
    inter_s = pd.Series([x[6]-layer_sim[model_type][layer] for x in van_sims if x[1]>x[2] and x[5] != 1])

    length = len(sense_s)
    mu_sense = round(mean(sense_s),2)
    sd_sense = round(stdev(sense_s),2)
    mu_inter = round(mean(inter_s),2)
    sd_inter = round(stdev(inter_s),2)

    # An "interface" to matplotlib.axes.Axes.hist() method
    w = 0.01
    n0 = math.ceil((self_s.max() - self_s.min())/w)
    n1 = math.ceil((sense_s.max() - sense_s.min())/w)
    n2 = math.ceil((inter_s.max() - inter_s.min())/w)


    fig, ax = plt.subplots(figsize = (15,8))
    # ax.hist(self_s, bins = n0, color='red', alpha=0.4, rwidth=0.85, label = 'self sim')
    # self_s.plot.kde(color = 'red', ax=ax, secondary_y=True, alpha = 1)
    ax.hist(sense_s, bins = n1, color='green', alpha=0.2, rwidth=0.85, label = 'sense similarity')
    sense_s.plot.kde(color = 'green', ax=ax, secondary_y=True, alpha = 1)
    ax.hist(inter_s, bins = n2, color='blue', alpha=0.2, rwidth=0.85, label = 'inter similarity')
    inter_s.plot.kde(color = 'blue', ax=ax, secondary_y=True, alpha = 1)
    # ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, 1])
    ax.legend()
#     plt.show()
    plt.savefig('/media/disk4/context_div/plots/'+str(model_type)+'/sense_vanilla_adjusted'+str(layer)+'.png')
    plt.clf()
    
    
    ##retrofitted

    self_s = pd.Series([x[4] for x in van_r_sims if x[1]>x[2] and x[5] != 1])
    sense_s = pd.Series([x[5] for x in van_r_sims if x[1]>x[2] and x[5] != 1])
    inter_s = pd.Series([x[6] for x in van_r_sims if x[1]>x[2] and x[5] != 1])

    length = len(sense_s)
    mu_r_sense = round(mean(sense_s),2)
    sd_r_sense = round(stdev(sense_s),2)
    mu_r_inter = round(mean(inter_s),2)
    sd_r_inter = round(stdev(inter_s),2)

    # An "interface" to matplotlib.axes.Axes.hist() method
    w = 0.01
    n0 = math.ceil((self_s.max() - self_s.min())/w)
    n1 = math.ceil((sense_s.max() - sense_s.min())/w)
    n2 = math.ceil((inter_s.max() - inter_s.min())/w)


    fig, ax = plt.subplots(figsize = (15,8))
    # ax.hist(self_s, bins = n0, color='red', alpha=0.4, rwidth=0.85, label = 'self sim')
    # self_s.plot.kde(color = 'red', ax=ax, secondary_y=True, alpha = 1)
    ax.hist(sense_s, bins = n1, color='green', alpha=0.2, rwidth=0.85, label = 'sense similarity')
    sense_s.plot.kde(color = 'green', ax=ax, secondary_y=True, alpha = 1)
    ax.hist(inter_s, bins = n2, color='blue', alpha=0.2, rwidth=0.85, label = 'inter similarity')
    inter_s.plot.kde(color = 'blue', ax=ax, secondary_y=True, alpha = 1)
    # ax.grid(axis='y', alpha=0.75)
    ax.set_xlabel('Cosine Similarity')
    ax.set_ylabel('Frequency')
    ax.set_xlim([0, 1])
    ax.legend()
#     plt.show()
    plt.savefig('/media/disk4/context_div/plots/'+str(model_type)+'/sense_retro'+str(layer)+'.png')
    plt.clf()
    
    layer_stats_vanilla.append((layer, mu_sense, sd_sense, mu_inter, sd_inter))
    layer_stats_retro.append((layer, mu_r_sense, sd_r_sense, mu_r_inter, sd_r_inter))
    
print(layer_stats_vanilla)
print(layer_stats_retro) 
print('done!')
    
    
    
    
    
    
    #proportion of words with self sim > inter sim
    
    #how different (larger or smaller) is self sim vs inter sim
    prop_van = round(len([1 for x in van_sims if x[5]>x[6] and 
                         x[1]>x[2]])/len([1 for x in van_sims if x[1]>x[2]]),3)
    
    prop_retro = round(len([1 for x in van_r_sims if x[5]>x[6] and 
                         x[1]>x[2]])/len([1 for x in van_r_sims if x[1]>x[2]]),3)
    
    delta_van = np.mean([(x[5]-x[6]) for x in van_sims])
        
    delta_retro = np.mean([(x[5]-x[6]) for x in van_r_sims])
    
    layer_stats.append((layer, prop_van, delta_van, prop_retro, delta_retro))
    
print(layer_stats)