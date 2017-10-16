import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_n_states = self.n_constant
        return self.base_model(best_n_states)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        """ select the best model for self.this_word based on
        average log Likelihood of cross-validation folds 
        for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_n_states, best_likelihood = None, None
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
                sum_likelihood = 0
                n_folds = 0

                split_method = KFold()
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    X, lengths = combine_sequences(cv_train_idx,self.sequences)
                    logL = self.base_model(n_states).score(X, lengths)
                    sum_likelihood += logL
                    n_folds += 1

                current_likelihood = sum_likelihood/n_folds

                if best_likelihood is None or best_likelihood < current_likelihood:
                    best_n_states = n_states
                    best_likelihood = current_likelihood
            except:
                pass

        if best_n_states is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_n_states)


class SelectorBIC(ModelSelector):
    """ select best model based on Bayesian Information Criterion(BIC) score:
    BIC = -2 * logL + p * logN
      logL: log-likelihood of the fitted model
      p:    # of parameters of the Gaussian HMM model
      N:    # of data points (samples)
    The lower BIC the better.
    
    Source:
        http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdfs
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_n_states, best_BIC = None, None
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
                logL = self.base_model(n_states).score(self.X, self.lengths)
                '''
                The parameters of a hidden Markov model are of two types, transition parameters and emission parameters (also known as output probabilities).

                Transition parameters:
                For each of the n possible states that a hidden variable at time t can be in, there is a transition probability from this state to each of the n possible states at time t+1, for a total of n**2 transition probabilities. 
                Note that the probabilities for transitions from any given state must sum to 1. Because any one transition probability can be determined once the others are known, there are a total of n x (n-1) transition parameters.
                --> n_states * (n_states-1)

                Emission parameters:
                For each of the n possible states, there is a set of emission probabilities governing the Gaussian distribution of the observed variable (feature) at a particular time given the state of the hidden variable at that time.
                Each Gaussian distribution of each observed variable (feature) has 2 parameters: a mean and a variance. We have 4 features, 2 for each hand (eg: right/left X/Y positions). 
                --> 2 * num_features * n_states

                Source: Hidden Markov model: Architecture (https://en.wikipedia.org/wiki/Hidden_Markov_model#Architecture)
                '''
                num_features = len(self.X[0])
                p = n_states * (n_states-1) + 2 * num_features * n_states
                logN = np.log(len(self.X))
                current_BIC = -2 * logL + p * logN

                if best_BIC is None or best_BIC > current_BIC:
                    best_n_states = n_states
                    best_BIC = current_BIC
            except:
                pass

        if best_n_states is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_n_states)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion (DIC) score:
    DIC = log(P(X(i))) - 1/(M-1)SUM(log(P(X(all but i))))
      M:                    # of words
      log(P(X(i))):         log-likelihood of fitted model for current word
      log(P(X(all but i))): log-likelihoods of fitted model for all the other words
    The higher DIC the better.

    Sources:
        Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
        Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
            https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    '''

    def select(self):
        """ select the best model for self.this_word based on
        DIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_n_states, best_DIC = None, None
        for n_states in range(self.min_n_components, self.max_n_components+1):
            try:
                M = len(self.words)
                log_P_X_i = self.base_model(n_states).score(self.X, self.lengths)
                
                SUM_log_P_X_all_but_i = 0
                words = list(self.words.keys())
                words.remove(self.this_word)               
                for word in words:
                    try:
                        model_selector_all_but_i = ModelSelector(self.words, self.hwords, word, self.n_constant, self.min_n_components, self.max_n_components, self.random_state, self.verbose)
                        log_P_X_all_but_i = model_selector_all_but_i.base_model(n_states).score(model_selector_all_but_i.X, model_selector_all_but_i.lengths)
                        SUM_log_P_X_all_but_i += log_P_X_all_but_i
                    except:
                        M -= 1
                
                current_DIC = log_P_X_i - 1/(M-1) * SUM_log_P_X_all_but_i

                if best_DIC is None or best_DIC < current_DIC:
                    best_n_states = n_states
                    best_DIC = current_DIC
            except:
                pass

        if best_n_states is None:
            return self.base_model(self.n_constant)
        else:
            return self.base_model(best_n_states)