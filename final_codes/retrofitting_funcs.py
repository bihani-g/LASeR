import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tqdm
from tqdm import tqdm
from scipy.spatial import distance
import sklearn


##getting embedding data
def get_embs(data_type, layer, model):
#     os.chdir('/media/disk4/context_div/emb_data/'+model+'/layer_'+str(layer)+'/')
    
    os.chdir('/nfs/gpusan01/gbihani/context_div/emb_data/'+model+'/layer_'+str(layer)+'/')

    data = []
    with open(str(data_type)+'_embs.pkl', 'rb') as f:
        while True:
            try:
                a = pickle.load(f)
            except EOFError:
                break
            else:
                data.append(a)
    return data


##getting retrofitted embedding data
def get_retro_embs(data_type, layer, model, d):
#     os.chdir('/media/disk4/context_div/emb_data/'+model+'/layer_'+str(layer)+'/')
    os.chdir('/nfs/gpusan01/gbihani/context_div/emb_data/'+model+'/layer_'+str(layer)+'/')

    with open(str(data_type)+'_'+str(d)+'_retro_embs.pkl', 'rb') as f:
        data = pickle.load(f)
        
    return data

#centering
def mean_centering(embs):
    emb_all = [np.array(x[1]) for x in embs]
    centered = np.mean(emb_all, axis=0)
    emb_c = emb_all - centered
    return emb_c

#format vector
def format_vector(embs, vanilla):
    vector_list = embs
    id_ = [x[0][0] for x in vanilla]
    word_ = [x[0][1] for x in vanilla]
    sense_ = [x[0][2] for x in vanilla]
    
    return list(zip(zip(id_, word_, sense_), vector_list))

#find var in top d pca dimensions
def top_d_var(embs,d):
    emb_all = [np.array(x[1]) for x in embs]
    pca = PCA(n_components=d)
    pca.fit_transform(emb_all)
    exp_var = [round(x*100,2) for x in pca.explained_variance_ratio_]
#     print([round(x*100,2) for x in pca.explained_variance_ratio_])
    
    return exp_var
    
##remove top d pca dimensions

def dim_removal(embs, d):
    embeds = [x[1] for x in embs]    
    mu = np.mean(embeds, axis=0)
    pca = sklearn.decomposition.PCA()
    pca.fit(embeds)
    Xhat = np.dot(pca.transform(embeds)[:,d:768], pca.components_[d:768,:])
    Xhat += mu
    embeds_ = Xhat
    return embeds_

##retrofitting function
def retrofit(embs):
    vector_list = []

    for i in tqdm(range(len(embs))):
        sense = embs[i][0][2]
        neighbours = [x[1] for x in embs if x[0][2] == sense]
        n_i = len(neighbours)
        
        for j in range(n_i):
            q_i_cap = embs[i][1]
            n_i_q_i_cap = n_i * q_i_cap
            for k in range(n_i):
                q_j = neighbours[k]
                n_i_q_i_cap += q_j
            
            numerator_i = n_i_q_i_cap
            denominator_i = 2*n_i
            q_i_new = numerator_i/denominator_i
            
        vector_list.append(q_i_new)
    
    return vector_list