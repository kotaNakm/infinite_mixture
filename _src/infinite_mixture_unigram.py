import pandas as pd
import numpy as np
from tqdm import tqdm
import numba

ZERO = 1.0e-10
MAX_K = 100

class infinite_mixture_unigram(object):
    def __init__(
        self,
        alpha,
        max_iter,
        random_state,
        verbose,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def init_status(self, X):
        """ Initialize Counters for matrix """
        self.n_events = len(X)
        self.n_dims = X.max().values + 1
        self.hist_k_num_all = np.zeros((self.max_iter))
        # max_k = self.n_events

        # for dataframe
        self.counterM = np.zeros((self.n_dims[1], MAX_K), dtype=int)
        self.counterK = np.zeros(MAX_K, dtype=int)
        # document aware total word counts 
        # self.counterA = np.zeros(self.n_dims[0], dtype=int)
        # Asum = X.groupby(X.columns[0]).size()
        # self.counterA[Asum.index] = Asum.values

        # numpy
        # self.counterM = np.zeros((X.shape[1], max_k), dtype=int)
        # self.counterK = np.zeros(max_k, dtype=int)
        # # document aware total word counts 
        # self.counterA = np.sum(X, axis=0)

        self.assignment =  np.full(self.n_events, -1, dtype=int)
        self.k_index = np.zeros(MAX_K,dtype=int)
        self.k_index[0] = 1
        self.k_last = 0

    def fit(self, X):
        """
        X: given numpy array (documents x words)
        """
        self.init_status(X)

        for iter_ in range(self.max_iter):
            self.assignment, self.k_last, hist_k_num = self.sample_topic(X.to_numpy(),)
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
        return _gibbs_sampling_CRP(     
            X,
            self.assignment,
            self.k_index,
            self.k_last,
            self.counterM,
            self.counterK,
            self.alpha,
            self.random_state,
            )

    def compute_vector(self,):
        k_orders = np.arange(len(self.k_index))
        activated_topics = k_orders[self.k_index==1]
        self.topic_dist = self.counterK[activated_topics] / self.counterK[activated_topics].sum()
        topic_aware_words_dist = self.counterM[:,activated_topics] / np.sum(self.counterM[:,activated_topics],axis=0)
        self.components_ = topic_aware_words_dist

@numba.jit(nopython=True)
def _gibbs_sampling_CRP(
    X,
    Z,
    k_index,
    k_last,
    counterM,
    counterK,
    alpha,
    random_state):

    np.random.seed(random_state)
    """
    X: event matrix 
    Z: topic assignments of the previous sampling
    """

    hist_k_num = np.full(len(Z),-1)
    max_k = len(counterK) 

    # for e, x in tqdm(enumerate(X),total=len(Z)):
    for e, x in enumerate(X):
        doc_ind = x[0]
        word_ind = x[1]
        pre_topic = Z[e]
        if not pre_topic == -1:
            counterK[pre_topic] -= 1
            counterM[word_ind, pre_topic] -=1
            if counterK[pre_topic] == 0:
                k_index[pre_topic] =0
                # np.delete(k_index, pre_topic)

        """ compute posterior distribution """
        posts = np.zeros(max_k, dtype=np.float64)
        add_topic = k_last + 1
        cm = counterM.astype(np.float64)
        posts = np.sum(cm,axis=0)
        posts[add_topic]  = alpha
        posts = posts / (posts.sum() + ZERO)

        try:
            new_topic = np.argmax(np.random.multinomial(1, posts))
        except:
            print("cannot calc assignment posterior:")
            # print(posts)
            # print(k_index)
            # print(posts.sum())
            # print(add_topic)
            # print(np.max(k_index) + 1)
            # print(len(posts))
            # print(counterM[word_ind])

        if new_topic == add_topic:
            k_index[add_topic] = 1
            k_last += 1

        Z[e] = new_topic
        counterK[new_topic] += 1
        counterM[word_ind, new_topic] += 1
    hist_k_num = k_index.sum() 

    return Z, k_last, hist_k_num