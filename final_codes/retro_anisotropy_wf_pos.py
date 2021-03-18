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
plt.style.use('science')
from random import sample

from retrofitting_funcs import get_embs, get_retro_embs, mean_centering, format_vector, top_d_var, dim_removal, retrofit
from sim_funcs import get_self_sim, get_avg_sense_sim, get_inter_sim, get_wf, get_sf, sim_combine


datasets = ['semeval15_13', 'senseval3task1', 'senseval2', 'semeval13_12', 'semeval07_7']
# 'raganato_ALL',  'senseval3task6', 'senseval3task6_train']

os.chdir('/media/disk4/context_div/WSD_data/mydata/')

with open('nouns.pkl', 'rb') as f:
    nouns = pickle.load(f)
    
with open('verbs.pkl', 'rb') as f:
    verbs = pickle.load(f)
    
with open('adjectives.pkl', 'rb') as f:
    adjectives = pickle.load(f)
    
#all words (do PCs capture word freq info?)
model_type = sys.argv[1]

data = {}
van = {}
words = {}
van_vec = {}

for layer in range(13):
    for data_num in datasets:
        data[data_num] = get_retro_embs(data_num, layer, model_type, 1)
        van[data_num] = [((x[0][0], x[0][1], x[0][2]), x[1]) for x in data[data_num]]
        van_vec[data_num] = [x[1] for x in van[data_num]]

    wfs_ = []
    vans_ = []
    for data_num in datasets:
        vans_i = van[data_num]
        vans_ = vans_ + vans_i
    
    vans_filt_ = [x for x in vans_ if ((x[0][1] in [y1[0] for y1 in nouns]) or (x[0][1] in [y2[0] for y2 in verbs]) or
                                     (x[0][1] in [y3[0] for y3 in adjectives]))]
    
    vans_ = sample(vans_filt_, 1000) ##sampled for k
    wf_ = [x[0][1] for x in vans_]
    wfs_ = [(x[0][1], wf_.count(x[0][1])) for x in vans_]
    pos = []
    
    for i in vans_:
        if i[0][1] in [y1[0] for y1 in nouns]:
            pos.append('noun')
        elif i[0][1] in [y2[0] for y2 in verbs]:
            pos.append('verb')
        elif i[0][1] in [y3[0] for y3 in adjectives]:
            pos.append('adjective')

    print('For layer ', layer)
    print('words_all:', len(vans_), flush = True)
    print(top_d_var(vans_, 10))
    print(round(sum(top_d_var(vans_, 10)), 3))
        
    emb_all = [np.array(x[1]) for x in vans_]
    count = 0
    sim_sum = 0
    for i in tqdm(range(len(emb_all))):
        for j in range(len(emb_all)):
            if i!=j:
                sim = 1- distance.cosine(emb_all[i],
                                     emb_all[j])
                sim_sum = sim_sum + sim
                count = count + 1
            
    print('avg. sim between all words:', round((sim_sum/count),3))
        
    pca = PCA(n_components=2)
    Principal_components = pca.fit_transform(emb_all)
    pca_df = pd.DataFrame(data = Principal_components, columns = ['PC 1', 'PC 2'])
    pca_df['wf'] = [x[1] for x in wfs_]
    pca_df['word'] = [x[0] for x in wfs_]
    pca_df['pos'] = pos
        
    cm = plt.cm.get_cmap('rainbow')
    z = [x[1] for x in wfs_]
    min_ = min([x[1] for x in wfs_])
    max_ = max([x[1] for x in wfs_])
    
#     outpath = "/media/disk4/context_div/plots/"
        
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel(r'$\alpha_{1}$', fontsize = 20)
    ax.set_ylabel(r'$\alpha_{2}$', fontsize = 20)
    ax.set_title('PCs vs. Frequency for layer '+str(layer), fontsize = 15)
    sc = ax.scatter(pca_df['PC 1'], pca_df['PC 2'], c=z, vmin=min_, vmax=max_, s=30, cmap=cm, alpha = 0.8)
    plt.axhline(y = 0.0, color = 'black', linestyle = ':' )
    plt.axvline(x = 0.0, color = 'black', linestyle = ':')
    plt.colorbar(sc)
    
    plt.savefig('/media/disk4/context_div/plots/'+str(model_type)+'/wf_retro_aniso_'+str(layer)+'.png')
    plt.clf()
    
    groups = pca_df.groupby("pos")
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    
    colors = {'noun':'mediumblue', 'verb':'springgreen', 'adjective':'magenta', 'other':'grey'}
    
    for name, group in groups:
        ax.plot(group["PC 1"], group["PC 2"], marker="o", linestyle="", label=name, color = colors[name],
               alpha = 0.8)
    ax.set_xlabel(r'$\alpha_{1}$', fontsize = 20)
    ax.set_ylabel(r'$\alpha_{2}$', fontsize = 20)
    ax.set_title('PCs vs. Frequency for layer '+str(layer), fontsize = 15)
    plt.legend(frameon=True)
    plt.axhline(y = 0.0, color = 'black', linestyle = ':' )
    plt.axvline(x = 0.0, color = 'black', linestyle = ':')
        
    plt.savefig('/media/disk4/context_div/plots/'+str(model_type)+'/pos_retro_aniso_'+str(layer)+'.png')
    plt.clf()
