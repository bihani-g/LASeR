import os
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tqdm
from tqdm import tqdm
from scipy.spatial import distance


#self sim
def get_self_sim(word, data):
    dat = [x[1] for x in data if x[0][1] == word]
    count = 0
    sim_sum = 0
    
    if len(dat)>1:
        for i in range(len(dat)):
            for j in range(len(dat)):
                if j!=i:
                    sim = 1- distance.cosine(dat[i],
                                     dat[j])
                    sim_sum = sim_sum + sim
                    count = count + 1
                    
    else:
        count = 1
        sim_sum = 1
    
    return round(sim_sum/count,3)


#sense sim
def get_avg_sense_sim(word, data):
    dat = [x for x in data if x[0][1] == word]
    w_sense_unq = list(set([x[0][2] for x in dat]))
    if len(w_sense_unq) > 1:
        sense_sim = []
        for sense in w_sense_unq:
            sensim_sum = 0
            c = 0
            same_sense_embs = [x[1] for x in dat if x[0][2] == sense]
            
            if len(same_sense_embs)>1: #multiple entries for that sense
                for i in range(len(same_sense_embs)):
                    for j in range(len(same_sense_embs)):
                        if i!=j:
                            sim = 1- distance.cosine(same_sense_embs[i],same_sense_embs[j])
                            sensim_sum = sensim_sum + sim
                            c = c + 1
            else: #only one entry for that sense
                sensim_sum = 1
                c = 1
            sense_sim.append(round(sensim_sum/c,3))
    
        avg_sen_sim = 0
    
        for s in sense_sim:
            avg_sen_sim = avg_sen_sim + s
        
        final_sen_sim = round(avg_sen_sim/len(sense_sim),3)

#         print(list(zip(w_sense_unq, sense_sim))) #get this for getting sense-level similarities for each word
    
    else: #sense sim is self similarity, as there is only one sense
        sim_sum = 0
        count = 0
        dat_ = [x[1] for x in data if x[0][0] == word]
        
        if len(dat) > 1: #if only one sense but multiple occurences
            for i in range(len(dat_)):
                for j in range(len(dat_)):
                    if j!=i:
                        sim = 1- distance.cosine(dat_[i],
                                     dat_[j])
                        sim_sum = sim_sum + sim
                        count = count + 1
        else: # only one sense and one occurence
            sim_sum = 1
            count = 1
            
        final_sen_sim = round(sim_sum/count,3)
        

        
#         print(w_sense_unq, final_sen_sim)
    
    return final_sen_sim
#     return list(zip(w_sense_unq, sense_sim))


#inter sim
def get_inter_sim(word, data):
    count = 0
    sim_sum = 0
    dat = [x for x in data if x[0][1] == word]
    if len(dat)>1:
        w_sense_unq = list(set([x[0][2] for x in dat]))
        if len(w_sense_unq) > 1:
            for i in range(len(dat)):
                for j in range(len(dat)):
                    if i!=j and dat[i][0][2] != dat[j][0][2]:
                        sim = 1- distance.cosine(dat[i][1],
                                     dat[j][1])
                        sim_sum = sim_sum + sim
                        count = count + 1

                    
        else: #inter sim is self similarity, as there is only one word per sense
            sim_sum = 0
            count = 0
            dat_ = [x[1] for x in data if x[0][1] == word]
            for i in range(len(dat_)):
                for j in range(len(dat_)):
                    if j!=i:
                        sim = 1- distance.cosine(dat_[i],
                                     dat_[j])
                        sim_sum = sim_sum + sim
                        count = count + 1
                        
    else:
        count = 1
        sim_sum = 1
        
                   
    return round(sim_sum/count,3)

#get word freq in data
def get_wf(word, data):
    count_ = len([x for x in data if x[0][1] == word])
    return count_

#get sense frequency in data
def get_sf(word, data):
    dat = [x[0][2] for x in data if x[0][1] == word]
    senses = len(list(set(dat)))   
    return senses

##for each word, we calculate (non repeatable)
# word - frequency - sense - self_sim - avg. sense_sim - inter_sim
def sim_combine(words, data):
    overall_ = [] #word freq, sense freq, self_sim, avg. sense_sim, inter_sim   
    
    for word in tqdm(words):
#         print(word)
        word_l = []
        wf_ = []
        sf_ = []
        self_ = []
        sense0_ = []
        sense1_ = []
        inter_ = []
        sense_sep = get_sense_sim(word, data)
        for i in range(len(sense_sep)):
            word_l.append(word)
            wf_.append(get_wf(word, data))
            sf_.append(get_sf(word, data))
            self_.append(get_self_sim(word, data))
#             sense_.append(get_avg_sense_sim(word, data))
            sense0_.append(sense_sep[i][0])
            sense1_.append(sense_sep[i][1])
            inter_.append(get_inter_sim(word, data))
        
        all_ = list(zip(word_l, wf_, sf_, sense0_, self_, sense1_, inter_))
        overall_.append(all_)
    
    overall_ = [y for x in overall_ for y in x]

    multi_ = [x for x in overall_ if x[1] > 1]
    multi_s_ = [x for x in multi_ if x[2] > 1]
        
    return multi_s_

#sense sim
def get_sense_sim(word, data):
    dat = [x for x in data if x[0][1] == word]
    w_sense_unq = list(set([x[0][2] for x in dat]))
#     print(w_sense_unq)
    if len(w_sense_unq) > 1:
        sense_sim = []
        for sense in w_sense_unq:
#             print(sense)
            sensim_sum = 0
            c = 0
            same_sense_embs = [x[1] for x in dat if x[0][2] == sense]
            
            if len(same_sense_embs)>1: #multiple entries for that sense
                for i in range(len(same_sense_embs)):
                    for j in range(len(same_sense_embs)):
                        if i!=j:
                            sim = 1- distance.cosine(same_sense_embs[i],same_sense_embs[j])
                            sensim_sum = sensim_sum + sim
                            c = c + 1
                                             
            else: #only one entry for that sense
                sensim_sum = 1
                c = 1
                
            sense_sim.append(round(sensim_sum/c,3))
#         print(sense_sim)
            
        combined_sim = list(zip(w_sense_unq, sense_sim))


#         print(list(zip(w_sense_unq, sense_sim))) #get this for getting sense-level similarities for each word
    
    else: #sense sim is self similarity, as there is only one sense
        sense_sim = []
        sim_sum = 0
        count = 0
        dat_ = [x[1] for x in data if x[0][0] == word]
        
        if len(dat_) > 1: #if only one sense but multiple occurences
            for i in range(len(dat_)):
                for j in range(len(dat_)):
                    if j!=i:
                        sim = 1- distance.cosine(dat_[i],
                                     dat_[j])
                        sim_sum = sim_sum + sim
                        count = count + 1
            
#             sense_sim.append(round(sensim_sum/c,3))
            
            combined_sim = [(w_sense_unq, round(sim_sum/count,3))]
                        
        else: # only one sense and one occurence
            sim_sum = 1
            count = 1
              
            combined_sim = [(w_sense_unq, round(sim_sum/count,3))]
        

        
#         print(w_sense_unq, final_sen_sim)
    
    return combined_sim
#     return list(zip(w_sense_unq, sense_sim))



