import pandas as pd
import numpy as np
from tqdm import tqdm
import numba
import math
from scipy.special import gamma, digamma, gammaln 


ZERO = 1.0e-10
MAX_K = 100

class infinite_mixture_unigram(object):
    def __init__(
        self,
        alpha,
        beta,
        max_iter,
        random_state,
        verbose,
    ):
        
        self.alpha = alpha
        self.beta = beta
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def init_status(self, X):
        """ Initialize Counters for matrix """
        self.n_events = len(X)
        self.n_dims = X.max().values + 1
        self.hist_k_num_all = np.zeros((self.max_iter))
        # max_k = self.n_events


        self.counter_docs = np.zeros((self.n_dims[0], MAX_K), dtype=np.float32)
        self.counter_words = np.zeros((self.n_dims[1], MAX_K), dtype=np.float32)
        # self.counterK = np.zeros(MAX_K, dtype=int)
        # document aware total word counts 
        # self.counterA = np.zeros(self.n_dims[0], dtype=int)
        # Asum = X.groupby(X.columns[0]).size()
        # self.counterA[Asum.index] = Asum.values
        # numpy
        # self.counterM = np.zeros((X.shape[1], max_k), dtype=int)
        # self.counterK = np.zeros(max_k, dtype=int)
        # # document aware total word counts 
        # self.counterA = np.sum(X, axis=0)

        self.assignment =  np.full(self.n_dims[0], -1, dtype=int) # document-wise  
        self.k_index = np.zeros(MAX_K,dtype=int)
        self.k_index[0] = 1
        self.k_last = 0

    def fit(self, X):
        """
        X: given numpy array (documents x words)
        """
        self.init_status(X)
        # for dataframe
        self.bow = pd.crosstab(X[X.columns[0]],X[X.columns[1]]).values.astype(np.float32)

        for iter_ in range(self.max_iter):
            self.assignment, self.k_last, hist_k_num = self.sample_topic(self.bow)
            self.hist_k_num_all[iter_] = hist_k_num
            print(f"ITERATION: {iter_+1}/{self.max_iter}")
            if self.verbose:
                print(f"# of topics: {hist_k_num}")

        self.compute_vector()

        if self.k_last == MAX_K:
            print("# of topic have reached maximum number. Change \"MAX_K\"")
            exit() 

    def fit_transform(self, X):
        self.fit(X)
        
        return self.topic_dist

    def sample_topic(self, X):
        # X: bow
        return _gibbs_sampling_CRP(     
            X,
            self.assignment,
            self.k_index,
            self.k_last,
            self.bow,
            self.counter_docs,
            self.counter_words,
            self.alpha,
            self.beta,
            self.random_state,
            )

    def compute_vector(self,):
        k_orders = np.arange(len(self.k_index))
        activated_topics = k_orders[self.k_index==1]
        self.topic_dist = self.counterK[activated_topics] / self.counterK[activated_topics].sum()
        topic_aware_words_dist = self.counterM[:,activated_topics] / np.sum(self.counterM[:,activated_topics],axis=0)
        self.components_ = topic_aware_words_dist

# @numba.jit(nopython=True)
def _gibbs_sampling_CRP(
    X,
    Z,
    k_index,
    k_last,
    bow,
    counter_docs,
    counter_words,
    alpha,
    beta,
    random_state):

    np.random.seed(random_state)
    """
    X: event matrix (bag of words)
    Z: topic assignments of the previous sampling
    """
    nunique_words = float(bow.shape[1])
    hist_k_num = np.full(len(Z),-1)
    max_k=len(k_index)

    for doc_ind, doc_words_freq in tqdm(enumerate(X),total=X.shape[0]):
    # for doc_ind, doc_words_freq in enumerate(X):
        pre_topic = Z[doc_ind]
        doc_nwords = np.sum(doc_words_freq)
        if not pre_topic == -1:
            counter_docs[doc_ind, pre_topic] -=1
            for word_ind, word_freq in enumerate(doc_words_freq):
                if word_freq > 0:
                    counter_words[word_ind, pre_topic] -=1
            if np.sum(counter_docs,axis=0)[pre_topic] == 0:
                k_index[pre_topic] = 0


        """ compute posterior distribution """
        posts = np.zeros(max_k, dtype=np.float32)
        add_topic = k_last + 1        

        # a. calc post prob for existing topic 
        # posts = np.sum(counter_docs,axis=0)
        # posts *= (np.sum(counter_words,axis=0) + beta * nunique_words) / (np.sum(counter_words,axis=0) + doc_nwords + beta * nunique_words)
        posts = np.log(np.sum(counter_docs,axis=0) + ZERO)
        print(np.sum(counter_docs,axis=0))
        posts += gammaln(np.sum(counter_words,axis=0) + beta * nunique_words) - gammaln(np.sum(counter_words,axis=0) + doc_nwords + beta * nunique_words) + 0
        print(posts)
        for word_ind, w_freq in enumerate(doc_words_freq):
            if w_freq > 0:
                # posts *= gamma(counter_words[word_ind] + w_freq + beta) / gamma(counter_words[word_ind]+ beta)
                posts += gammaln(counter_words[word_ind] + w_freq + beta) - gammaln(counter_words[word_ind]+ beta)
        print(posts)
        # b. calc post prob for new topic 
        # posts[add_topic]  = alpha
        posts[add_topic]  = np.log(alpha)
        # posts[add_topic]  *= gamma(beta * nunique_words) / gamma(doc_nwords + beta * nunique_words)
        posts[add_topic]  += gammaln(beta * nunique_words) - gammaln(doc_nwords + beta * nunique_words)
        for word_ind, w_freq in enumerate(doc_words_freq):
            if w_freq > 0:
                # posts[add_topic] *= gamma(w_freq + beta) / gamma(beta)
                posts[add_topic] += gammaln(w_freq + beta) - gammaln(beta)
        print(posts)

        # draw
        posts[posts < 0] = 0
        posts = posts / (posts.sum() + ZERO)
        try:
            new_topic = np.argmax(np.random.multinomial(1, posts))
        except:
            print("cannot calc assignment posterior:")
            print(posts)
            print(k_index)
            print(posts.sum())
            print(add_topic)
            print(np.max(k_index) + 1)
            print(len(posts))
            print(counter_words[word_ind])

        if new_topic == add_topic:
            k_index[add_topic] = 1
            k_last += 1

        Z[doc_ind] = new_topic
        counter_docs[doc_ind, new_topic] += 1
        for word_ind, word_freq in enumerate(doc_words_freq):
            if word_freq > 0:
                counter_words[word_ind, new_topic] += 1
    
    hist_k_num = k_index.sum() 

    return Z, k_last, hist_k_num