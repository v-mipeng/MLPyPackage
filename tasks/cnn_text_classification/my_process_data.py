import numpy as np
import cPickle
from collections import defaultdict
import sys, re
import pandas as pd
reload(sys)
sys.setdefaultencoding('utf-8')

def build_data_cv(data_folder, cv=10, clean_string=True):
    """
    Loads data and split into 10 folds.
    """
    revs = []
    trn_file = data_folder[0]
    test_file = data_folder[1]
    vocab = defaultdict(float)
    
    with open(trn_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append(line.strip('\t').split('\t')[1])
            #label = int(line.strip('\t').split('\t')[0])
            label = line.strip('\t').split('\t')[0]
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev)
            words = set(orig_rev.split())
#            print orig_rev
            for word in words:
                vocab[word] += 1
            datum  = {"y":label, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 1}
            revs.append(datum)

    with open(test_file, "rb") as f:
        for line in f:       
            rev = []
            rev.append((line.strip('\t').split('\t')[1]).encode('utf-8'))
            #label = int(line.strip('\t').split('\t')[0])
            label = line.strip('\t').split('\t')[0]
            print rev
            print label

            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev)
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            datum  = {"y":label, 
                      "text": orig_rev,                             
                      "num_words": len(orig_rev.split()),
                      "split": 0}
            revs.append(datum)
            break

    return revs, vocab
    
def get_W(word_vecs, k=100):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))            
    W[0] = np.zeros(k)
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 100x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, 'r') as f:
        for line in f:
            word = line.strip().split('\t')[0]
            embeding = line.strip().split('\t')[1].split()
            if word in vocab:
                #print embeding
                word_vecs[word] = np.asarray(embeding, dtype='float64')
    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)  

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string) 
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip() if TREC else string.strip().lower()

def clean_str_sst(string):
    """
    Tokenization/string cleaning for the SST dataset
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\s{2,}", " ", string)    
    return string.strip().lower()

if __name__=="__main__":    
    w2v_file = sys.argv[1]     
    data_folder = ["train.txt","test.txt"]    
    print "loading data...",        
    revs, vocab = build_data_cv(data_folder, cv=10, clean_string=False)
    max_l = np.max(pd.DataFrame(revs)["num_words"])
    print "data loaded!"
    print "number of sentences: " + str(len(revs))
    print "vocab size: " + str(len(vocab))
    print "max sentence length: " + str(max_l)
    print "loading word2vec vectors...",
    w2v = load_bin_vec(w2v_file, vocab)
    print "word2vec loaded!"
    print "num words already in word2vec: " + str(len(w2v))
    add_unknown_words(w2v, vocab)
    W, word_idx_map = get_W(w2v)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab)
    W2, _ = get_W(rand_vecs)
    cPickle.dump([revs, W, W2, word_idx_map, vocab], open("mr.p", "wb"))
    print "dataset created!"
    
