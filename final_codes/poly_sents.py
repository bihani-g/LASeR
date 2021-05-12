####polysemous word and sent id extraction


import os
import pickle
import numpy as np
import sys
import tqdm
from tqdm import tqdm

data_type = sys.argv[1]

def get_poly_sents(senses, word):
    num_senses = len(list(set([y[2] for y in [x for x in senses if x[1] == word]])))
    
    if num_senses > 1:
        sent_ids = [y[0] for y in [x for x in senses if x[1] == word]]
    
    else:
        sent_ids = ['None']
        
    return sent_ids


#get a list of polysemous words
os.chdir('../WSD_data/mydata/')

with open(str(data_type)+'_senses.pkl', 'rb') as f:
    senses = pickle.load(f)

words = list(set([x[1] for x in senses]))

poly_sents = []
for word in tqdm(words):
    poly_s = get_poly_sents(senses, word)
    
    if poly_s != ['None']:
        poly_sents.append((word, get_poly_sents(senses, word)))
        
new = [(y, x[0]) for x in poly_sents for y in x[1]]
new_set = list(set([x[0] for x in new]))

#get a list of sentences with polysemous words, each sentence id and polysemous words in it
sent_words = []
for sid in tqdm(new_set):
    words_all = [y[1] for y in new if y[0] == sid]
    sent_words.append((sid, words_all))
    
#get a list of polysemous words
os.chdir('../WSD_data/mydata/')

with open(str(data_type)+'_poly.pkl', 'wb') as f:
    pickle.dump(sent_words, f)
    
    
    
print('done!')
      
      
      