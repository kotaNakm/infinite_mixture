import pandas as pd
import numpy as np
from tqdm import tqdm
import numba
from scipy.special import gamma, digamma, gammaln 


MAX_K = 1000
ZERO = 1.0e-8

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

        self.counter_docs = np.zeros((self.n_dims[0], MAX_K), dtype=np.float64)
        self.counter_words = np.zeros((self.n_dims[1], MAX_K), dtype=np.float64)

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
        self.topic_dist = (np.sum(self.counter_docs[:,activated_topics],axis=0) + self.alpha) / (self.counter_docs[:,activated_topics].sum() + self.alpha * len(activated_topics))
        topic_aware_words_dist = (self.counter_words[:,activated_topics] + self.beta) / (np.sum(self.counter_words[:,activated_topics],axis=0) + self.beta * self.n_dims[1])
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
        posts = np.zeros(max_k, dtype=np.float64)
        add_topic = k_last + 1        

        # a. calc post prob for existing topic 
        # posts = np.sum(counter_docs,axis=0)
        # posts *= (np.sum(counter_words,axis=0) + beta * nunique_words) / (np.sum(counter_words,axis=0) + doc_nwords + beta * nunique_words)
        posts = np.log(np.sum(counter_docs,axis=0) + ZERO)
        posts += gammaln(np.sum(counter_words,axis=0) + beta * nunique_words) - gammaln(np.sum(counter_words,axis=0) + doc_nwords + beta * nunique_words)
        for word_ind, w_freq in enumerate(doc_words_freq):
            if w_freq > 0:
                # posts *= gamma(counter_words[word_ind] + w_freq + beta) / gamma(counter_words[word_ind]+ beta)
                posts += gammaln(counter_words[word_ind] + w_freq + beta) - gammaln(counter_words[word_ind]+ beta)
        # b. calc post prob for new topic 
        # posts[add_topic]  = alpha
        posts[add_topic]  = np.log(alpha + ZERO)
        # posts[add_topic]  *= gamma(beta * nunique_words) / gamma(doc_nwords + beta * nunique_words)
        posts[add_topic]  += gammaln(beta * nunique_words) - gammaln(doc_nwords + beta * nunique_words)
        for word_ind, w_freq in enumerate(doc_words_freq):
            if w_freq > 0:
                # posts[add_topic] *= gamma(w_freq + beta) / gamma(beta)
                posts[add_topic] += gammaln(w_freq + beta) - gammaln(beta)

        # draw
        try:
            # posts = (posts[:add_topic+1]+1) / ((posts[:add_topic+1]+1).sum())
            # posts = posts[:add_topic+1] / ((posts[:add_topic+1]+1).sum() + ZERO)
            posts = posts[:add_topic+1] / ((posts[:add_topic+1]+1).sum())
            new_topic = draw_one(posts[:add_topic+1])
            # new_topic = np.argmax(np.random.multinomial(1, posts[:add_topic+1]))
            # print(posts)
        except:
            exit()

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

@numba.jit(nopython=True)
def draw_one(posts):
    residual = np.random.uniform(0, np.sum(posts))
    return_sample = 0
    for sample, prob in enumerate(posts):
        residual -= prob                    
        if residual < 0.0:
            return_sample = sample
            break  
    return return_sample