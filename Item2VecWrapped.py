import random
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from gensim.models import Word2Vec

class Item2VecWrapped(BaseEstimator):
    def __init__(
        self,
        alpha=0.025,
        cbow_mean=1,
        epochs=5,
        hs=0,
        min_alpha=0.0001,
        min_count=1,
        negative=5,
        ns_exponent=-0.5,
        sample=0.001,
        seed=1,
        sg=0,
        vector_size=100,
        window=3,
        shrink_windows=True,
        topK=10,
        split_strategy="timeseries"
    ):
        self.alpha = alpha
        self.cbow_mean = cbow_mean
        self.epochs = epochs
        self.hs = hs
        self.min_alpha = min_alpha
        self.min_count = min_count
        self.negative = negative
        self.ns_exponent = ns_exponent
        self.sample = sample
        self.seed = seed
        self.sg = sg
        self.vector_size = vector_size
        self.window = window
        self.shrink_windows = shrink_windows
        self.topK = topK
        self.split_strategy = split_strategy

    def fit(self, X):
        """
        Fit method (sklearn estimators interface)
        The template is not respected since X does not have the shape (n_features, n_examples)
        """
        if (self.split_strategy=='timeseries'):
            X_len = [x for x in X if len(x)>=2]
            X_train = [x[:-1] for x in X_len]
        elif(self.split_strategy=='train_test_split'):
            X_train, X_test_unused = train_test_split(X, test_size=0.05, random_state=42)
        

        self.model = Word2Vec(
            sentences=X_train,
            alpha = self.alpha,
            cbow_mean = self.cbow_mean,
            epochs = self.epochs,
            hs = self.hs,
            min_alpha = self.min_alpha,
            min_count = self.min_count,
            negative = self.negative,
            ns_exponent = self.ns_exponent,
            sample = self.sample,
            seed = self.seed,
            sg = self.sg,
            vector_size = self.vector_size,
            window = self.window,
            shrink_windows = self.shrink_windows
        )
        return self

    def transform(self, word=None, norm=False):
        """
        Transform method (sklearn estimators interface)
        """
        check_is_fitted(self, "model")
        if word == None:
            X_transform_vec = self.model.wv.vectors
        else:
            X_transform_vec = self.model.wv[word]
        return X_transform_vec

    
    def predict(self, X_val):
        """
        Predict function (usefull for score function)
        Return top 10 most similar words by index
        """
        check_is_fitted(self, "model")
        Y_pred = []
        for word in X_val:
            temp_most_sim = self.model.wv.most_similar(
                positive=word, topn=self.topK
            )
            Y_pred.append(
                [self.model.wv.get_index(temp_most_sim[j][0]) for j in range(self.topK)]
            )
        return Y_pred

    def score(self, X):
        """
        Score metrics for the grid_search (do not call outside the class)
        """
        check_is_fitted(self, "model")
        X_predict_w = []
        Y_true_w = []
        
        if(self.split_strategy=='timeseries'):
            X_len = [x for x in X if len(x)>=2]
            X_test = [x[-2:] for x in X_len]

            X_test_clean = [x for x in X_test if(x[0] in self.get_vocabulary() and x[1] in self.get_vocabulary())]
            X_test_clean_random = random.choices(X_test_clean,k=1000)
            X_predict_w = [x[0] for x in X_test_clean_random]
            Y_true_w = [x[1] for x in X_test_clean_random]
        elif(self.split_strategy=='train_test_split'):
            X_train_unused, X_test = train_test_split(X, test_size=0.05, random_state=42)
            X_test_clean = []
            for sub_X_test in X_test:
                for i in range(len(sub_X_test)-1):
                    if((sub_X_test[i] in self.get_vocabulary()) and (sub_X_test[i+1] in self.get_vocabulary())):
                        X_test_clean.append([sub_X_test[i], sub_X_test[i+1]])
            X_test_clean_random = random.choices(X_test_clean,k=1000)
            X_predict_w = [x[0] for x in X_test_clean_random]
            Y_true_w = [x[1] for x in X_test_clean_random]

        Y_predict = self.predict(X_predict_w)
        Y_true = [self.model.wv.get_index(word) for word in Y_true_w]
        res = np.array([1 if (Y_true[i] in Y_predict[i]) else 0 for i in range(len(Y_true))])
        res = (np.sum(res))/len(Y_true)
        return res

    def score_Precision_at_K(self,X_test, Y_test):
        """
        Score metrics, use this instead of score method
        """
        check_is_fitted(self, "model")
        if(len(X_test)==len(Y_test)):
            X_Y_couple = [[X_test[i], Y_test[i]] for i in range(len(X_test))]
            X_Y_clean = [x for x in X_Y_couple if(x[0] in self.get_vocabulary() and x[1] in self.get_vocabulary())]
            X_w = [x[0] for x in X_Y_clean]
            Y_true_w = [x[1] for x in X_Y_clean]
            Y_predict = self.predict(X_w)
            Y_true = [self.model.wv.get_index(word) for word in Y_true_w]
            print(Y_predict)
            res = np.array([1 if (Y_true[i] in Y_predict[i]) else 0 for i in range(len(Y_true))])
            res = (np.sum(res))/len(Y_true)
        else:
            res="Arguments must have same length"
        return res


    def get_vocabulary(self):
        """
        Get vocabulary (unique words)
        """
        check_is_fitted(self, "model")
        return self.model.wv.index_to_key

    def get_index_word(self, word):
        """Get index of a word in the vocabulary (unique words)"""
        return self.model.wv.get_index(word)

    def get_params(self, deep=True):
        """Get model parameters"""
        get_param = dict()
        for attribute, value in self.__dict__.items():
            get_param.update({attribute: value})
        return get_param

    def set_params(self, **parameters):
        """Set model parameters"""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self