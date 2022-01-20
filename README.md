
# Item2Vec Wrapped

Gensim 4.x
1. Introduction
2. General informations
3. Methods Description
4. GridSearch and BayesSearch Usage
5. BayesSearch Example

# 1. Introduction

Prod2Vec or Item2Vec produces embedding for items in a latent space. The method is capable of inferring item-item relations even when user information is not available. It's based on NLP model Word2Vec. [Click here](https://arxiv.org/pdf/1603.04259.pdf#:~:text=Inspired%20by%20SGNS%2C%20we%20describe,user%20information%20is%20not%20available.) to know more

This project provide a class that encapsulates Item2Vec model ([word2vec](https://radimrehurek.com/gensim/models/word2vec.html) gensim model) as a [sklearn estimator](https://scikit-learn.org/stable/developers/develop.html).

It allows the simple and efficient use of the Item2Vec model by providing :
- metric to measure the performance of the model ([Precision@K](https://arxiv.org/pdf/0704.3359.pdf))
- compatibility with [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html) to find the optimal hyperparameters


# 2. General informations

!! Warning : Estimators template is not respected since X does not have the shape (n_features, n_examples)

**2.1 Input data format** 


```
X : list of string list. Each string is an item. Each list is the sequence of products purchased 
by a customer/while a session
```

```
X = [['prod_1', ..., 'prod_n'], ... ,['prod_1', ..., 'prod_n']]
```


**2.2 Train/Test split**

The train/test split is managed within the class. It is not necessary to split the data into train and test before fitting the model.

**2.3 Pipeline performance measurement**

- Training on subset of X
- Randomly sampled ((n-1)-th, n-th) pairs of items, disjoint from the training set
- Evaluate performance on NEP task (ie: Find the top 10 most similar to the (n-1)-th item and check if the (n-th) item is in this top 10)

# 3. Methods description

**3.1 Instanciation and init parameters**

Instanciation :
```
Item2VecWrapped(alpha=0.025, cbow_mean=1, epochs=5,hs=0, min_alpha=0.0001, min_count=1, negative=5, ns_exponent=-0.5, 
sample=0.001, seed=1, sg=0, vector_size=100, window=3, shrink_windows=True,topK=10, split_strategy="timeseries")
```

Word2Vec default Parameters (Gensim 4.x)
```
alpha=0.025, cbow_mean=1, epochs=5,hs=0, min_alpha=0.0001, min_count=1, negative=5, 
ns_exponent=-0.5, sample=0.001, seed=1, sg=0, vector_size=100, window=3, shrink_windows=True,
```

Added parameters :
```
topK=int, split_strategy=string
```
topK : most similar word to a given word. (10 by default)

split_strategy : "timeseries" or "train_test_split"


```
"timeseries" : Training set -> (item_1, ..., item_N-1)
               Test set -> (item_N-1, item_N)

"train_test_split" : Training set, Test set = train_test_split(X, test_size=0.05, random_state=42)
                     Create couple (item_N-1, item_N) from Test_test

```

**3.2 Fit method**

```
fit(X)
```
- Getting X_train data (depending on splitting strategy) to train the gensim Word2Vec model
- Train Word2Vec model on X_train

**3.3 Predict method**

```
predict(X)
```

- ```X``` is a word or a list of words
- Predict topK most similar words using cosine similarity.

```Return``` a list of list of topK words by index


**3.4 Score method (not using it outside the classe)**

```
score(X)
```

```X``` must be the same as the one provide to fit()

Designed for the GridSearchCV and BayesSearch. Use ```score_Precision_at_K(X_test,Y_test)``` instead.

Evaluate performance on Next Event Prediction using Precision@K

```Return``` : The score in pecentage of right prediction


**3.5 Score_Precision_at_K method**


```
score_Precision_at_K(X_test, Y_test)
```
Evaluate performance on Next Event Prediction using Precision@K


```X_test``` : list of items

```Y_test``` : list of items. Ground truth about the next item purchases just after X_test

```Return``` : The score in pecentage of right prediction


**3.6 Get_vocabulary method**

```
get_vocabulary()
```
```Return``` : list of vocabulary after the training.

```Word2Vec().fit(X).get_vocabulary()[idx]``` will return word at index idx.

**3.7 Get_index_word method**

```
get_index_word(word)
```
```Return``` : Index of the given word

# 4. GridSearch and BayesSearch Usage

Model instantation
```
my_model = Item2VecWrapped()
```

Hyperparameters definition
```
parameters = {'ns_exponent': [1, 0.5, -0.5, -1], 'alpha': [0.1, 0.3, 0.6, 0.9]}
```

Define Train and test indices for splitting. 
!! Test and train indices must be the same !! The split is managed internally
```
train_indices = [i for i in range(len(X))]
test_indices = [i for i in range(len(X))]

cv = [(train_indices, test_indices)]
```

Instantiate GridSearchCV
```
clf = GridSearchCV(my_model,parameters, cv=cv)
```

Fit the model and getting best parameters and best scores
```
clf.fit(X)

clf.best_params_

clf.best_score_
```

# 5. BayesSearch Example

```
!pip install scikit-optimize

from skopt.space import Integer
from skopt.space import Real
from skopt.space import Categorical
from skopt.utils import use_named_args
from skopt import BayesSearchCV

search_space = list()
search_space.append(Integer(3, 100, name='epochs', prior='log-uniform', base=2))
search_space.append(Integer(10, 500, name='vector_size', prior='log-uniform', base=2))
search_space.append(Real(0.01, 1, name='alpha', prior='uniform'))
search_space.append(Real(-1, 1, name='ns_exponent', prior='uniform'))
search_space.append(Integer(5, 50, name='negative', prior='uniform'))
search_space.append(Categorical([0, 1], name='sg'))
search_space.append(Real(0.00001, 0.01, name='sample', prior='uniform'))
search_space.append(Categorical([0, 1], name='cbow_mean'))
search_space.append(Integer(1,3, name='window', prior='uniform')) #mean of basket len is 1.54
search_space.append(Categorical([True, False], name='shrink_windows'))


params = {search_space[i].name : search_space[i] for i in range((len(search_space)))}

train_indices = [i for i in range(len(X))]  # indices for training
test_indices = [i for i in range(len(X))]  # indices for testing

cv = [(train_indices, test_indices)]


clf = BayesSearchCV(estimator=Item2VecWrapped(), search_spaces=params, n_jobs=-1, cv=cv)

clf.fit(X)

clf.best_params_

clf.best_score_

```


