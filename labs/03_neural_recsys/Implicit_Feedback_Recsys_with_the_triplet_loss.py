
# coding: utf-8

# # Triplet Loss for Implicit Feedback Neural Recommender Systems
# 
# The goal of this notebook is first to demonstrate how it is possible to build a bi-linear recommender system only using positive feedback data.
# 
# In a latter section we show that it is possible to train deeper architectures following the same design principles.
# 
# This notebook is inspired by Maciej Kula's [Recommendations in Keras using triplet loss](
# https://github.com/maciejkula/triplet_recommendations_keras). Contrary to Maciej we won't use the BPR loss but instead will introduce the more common margin-based comparator.
# 
# ## Loading the movielens-100k dataset
# 
# For the sake of computation time, we will only use the smallest variant of the movielens reviews dataset. Beware that the architectural choices and hyperparameters that work well on such a toy dataset will not necessarily be representative of the behavior when run on a more realistic dataset such as [Movielens 10M](https://grouplens.org/datasets/movielens/10m/) or the [Yahoo Songs dataset with 700M rating](https://webscope.sandbox.yahoo.com/catalog.php?datatype=r).

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path as op

from zipfile import ZipFile
try:
    from urllib.request import urlretrieve
except ImportError:  # Python 2 compat
    from urllib import urlretrieve


ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_FILENAME = ML_100K_URL.rsplit('/', 1)[1]
ML_100K_FOLDER = 'ml-100k'

if not op.exists(ML_100K_FILENAME):
    print('Downloading %s to %s...' % (ML_100K_URL, ML_100K_FILENAME))
    urlretrieve(ML_100K_URL, ML_100K_FILENAME)

if not op.exists(ML_100K_FOLDER):
    print('Extracting %s to %s...' % (ML_100K_FILENAME, ML_100K_FOLDER))
    ZipFile(ML_100K_FILENAME).extractall('.')


# In[ ]:


train_data = pd.read_csv(op.join(ML_100K_FOLDER, 'ua.base'), sep='\t',
                         names=["user_id", "item_id", "rating", "timestamp"])
data_test = pd.read_csv(op.join(ML_100K_FOLDER, 'ua.test'), sep='\t',
                        names=["user_id", "item_id", "rating", "timestamp"])

train_data.describe()


# In[ ]:


def extract_year(release_date):
    if hasattr(release_date, 'split'):
        components = release_date.split('-')
        if len(components) == 3:
            return int(components[2])
    # Missing value marker
    return 1920


m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
items = pd.read_csv(op.join(ML_100K_FOLDER, 'u.item'), sep='|',
                    names=m_cols, usecols=range(5), encoding='latin-1')
items['release_year'] = items['release_date'].map(extract_year)

train_data = pd.merge(train_data, items)
data_test = pd.merge(data_test, items)


# In[ ]:


train_data.head()


# In[ ]:


# data_test.describe()


# In[ ]:


max_user_id = max(train_data['user_id'].max(), data_test['user_id'].max())
max_item_id = max(train_data['item_id'].max(), data_test['item_id'].max())

n_users = max_user_id + 1
n_items = max_item_id + 1

print('n_users=%d, n_items=%d' % (n_users, n_items))


# ## Implicit feedback data
# 
# Consider ratings >= 4 as positive feed back and ignore the rest:

# In[ ]:


pos_data_train = train_data.query("rating >= 4")
pos_data_test = data_test.query("rating >= 4")


# Because the median rating is around 3.5, this cut will remove approximately half of the ratings from the datasets:

# In[ ]:


pos_data_train['rating'].count()


# In[ ]:


pos_data_test['rating'].count()


# ## The Triplet Loss
# 
# The following section demonstrates how to build a low-rank quadratic interaction model between users and items. The similarity score between a user and an item is defined by the unormalized dot products of their respective embeddings.
# 
# The matching scores can be use to rank items to recommend to a specific user.
# 
# Training of the model parameters is achieved by randomly sampling negative items not seen by a pre-selected anchor user. We want the model embedding matrices to be such that the similarity between the user vector and the negative vector is smaller than the similarity between the user vector and the positive item vector. Furthermore we use a margin to further move appart the negative from the anchor user.
# 
# Here is the architecture of such a triplet architecture. The triplet name comes from the fact that the loss to optimize is defined for triple `(anchor_user, positive_item, negative_item)`:
# 
# <img src="images/rec_archi_implicit_2.svg" style="width: 600px;" />
# 
# We call this model a triplet model with bi-linear interactions because the similarity between a user and an item is captured by a dot product of the first level embedding vectors. This is therefore not a deep architecture.

# In[ ]:


import tensorflow as tf


def identity_loss(y_true, y_pred):
    """Ignore y_true and return the mean of y_pred
    
    This is a hack to work-around the design of the Keras API that is
    not really suited to train networks with a triplet loss by default.
    """
    return tf.reduce_mean(y_pred + 0 * y_true)


def margin_comparator_loss(inputs, margin=1.):
    """Comparator loss for a pair of precomputed similarities
    
    If the inputs are cosine similarities, they each have range in
    (-1, 1), therefore their difference have range in (-2, 2). Using
    a margin of 1. can therefore make sense.

    If the input similarities are not normalized, it can be beneficial
    to use larger values for the margin of the comparator loss.
    """
    positive_pair_sim, negative_pair_sim = inputs
    return tf.maximum(negative_pair_sim - positive_pair_sim + margin, 0)


# Here is the actual code that builds the model(s) with shared weights. Note that here we use the cosine similarity instead of unormalized dot products (both seems to yield comparable results).

# In[ ]:


from keras.models import Model
from keras.layers import Embedding, Flatten, Input, Dense
from keras.layers import Lambda, Dot
from keras.regularizers import l2
from keras.layers.merge import dot, concatenate


def build_models(n_users, n_items, latent_dim=64, l2_reg=0):
    """Build a triplet model and its companion similarity model
    
    The triplet model is used to train the weights of the companion
    similarity model. The triplet model takes 1 user, 1 positive item
    (relative to the selected user) and one negative item and is
    trained with comparator loss.
    
    The similarity model takes one user and one item as input and return
    compatibility score (aka the match score).
    """
    # Common architectural components for the two models:
    # - symbolic input placeholders
    user_input = Input((1,), name='user_input')
    positive_item_input = Input((1,), name='positive_item_input')
    negative_item_input = Input((1,), name='negative_item_input')

    # - embeddings
    l2_reg = None if l2_reg == 0 else l2(l2_reg)
    user_layer = Embedding(n_users, latent_dim, input_length=1,
                           name='user_embedding', embeddings_regularizer=l2_reg)
    
    # The following embedding parameters will be shared to encode both
    # the positive and negative items.
    item_layer = Embedding(n_items, latent_dim, input_length=1,
                           name="item_embedding", embeddings_regularizer=l2_reg)

    user_embedding = Flatten()(user_layer(user_input))
    positive_item_embedding = Flatten()(item_layer(positive_item_input))
    negative_item_embedding = Flatten()(item_layer(negative_item_input))

    # - similarity computation between embeddings
    positive_similarity = Dot(name="positive_similarity",
                              axes=1, normalize=True)(
        [user_embedding, positive_item_embedding])
    negative_similarity = Dot(name="negative_similarity",
                              axes=1, normalize=True)(
        [user_embedding, negative_item_embedding])

    # The triplet network model, only used for training
    triplet_loss = Lambda(margin_comparator_loss,
                          name='comparator_loss',
                          output_shape=(1,))(
        [positive_similarity, negative_similarity])

    triplet_model = Model(inputs=[user_input,
                                  positive_item_input,
                                  negative_item_input],
                          outputs=triplet_loss)
    
    # The match-score model, only use at inference to rank items for a given
    # model: the model weights are shared with the triplet_model therefore
    # we do not need to train it and therefore we do not need to plug a loss
    # and an optimizer.
    match_model = Model(inputs=[user_input, positive_item_input],
                        outputs=positive_similarity)
    
    return triplet_model, match_model


triplet_model, match_model = build_models(n_users, n_items, latent_dim=64,
                                          l2_reg=1e-6)


# ### Exercise:
# 
# How many trainable parameters does each model. Count the shared parameters only once per model.

# In[ ]:


# print(match_model.summary())


# In[ ]:


# print(triplet_model.summary())


# In[ ]:


# %load solutions/triplet_parameter_count.py


# ## Quality of Ranked Recommendations
# 
# Now that we have a randomly initialized model we can start computing random recommendations. To assess their quality we do the following for each user:
# 
# - compute matching scores for items (except the movies that the user has already seen in the training set),
# - compare to the positive feedback actually collected on the test set using the ROC AUC ranking metric,
# - average ROC AUC scores across users to get the average performance of the recommender model on the test set.

# In[ ]:


from sklearn.metrics import roc_auc_score


def average_roc_auc(match_model, data_train, data_test):
    """Compute the ROC AUC for each user and average over users"""
    max_user_id = max(data_train['user_id'].max(), data_test['user_id'].max())
    max_item_id = max(data_train['item_id'].max(), data_test['item_id'].max())
    user_auc_scores = []
    for user_id in range(1, max_user_id + 1):
        pos_item_train = data_train[data_train['user_id'] == user_id]
        pos_item_test = data_test[data_test['user_id'] == user_id]
        
        # Consider all the items already seen in the training set
        all_item_ids = np.arange(1, max_item_id + 1)
        items_to_rank = np.setdiff1d(all_item_ids, pos_item_train['item_id'].values)
        
        # Ground truth: return 1 for each item positively present in the test set
        # and 0 otherwise.
        expected = np.in1d(items_to_rank, pos_item_test['item_id'].values)
        
        if np.sum(expected) >= 1:
            # At least one positive test value to rank
            repeated_user_id = np.empty_like(items_to_rank)
            repeated_user_id.fill(user_id)

            predicted = match_model.predict([repeated_user_id, items_to_rank],
                                            batch_size=4096)
            user_auc_scores.append(roc_auc_score(expected, predicted))

    return sum(user_auc_scores) / len(user_auc_scores)


# By default the model should make predictions that rank the items in random order. The **ROC AUC score** is a ranking score that represents the **expected value of correctly ordering uniformly sampled pairs of recommendations**.
# 
# A random (untrained) model should yield 0.50 ROC AUC on average. 

# In[ ]:


average_roc_auc(match_model, pos_data_train, pos_data_test)


# ## Training the Triplet Model
# 
# Let's now fit the parameters of the model by sampling triplets: for each user, select a movie in the positive feedback set of that user and randomly sample another movie to serve as negative item.
# 
# Note that this sampling scheme could be improved by removing items that are marked as positive in the data to remove some label noise. In practice this does not seem to be a problem though.

# In[ ]:


def sample_triplets(pos_data, max_item_id, random_seed=0):
    """Sample negatives at random"""
    rng = np.random.RandomState(random_seed)
    user_ids = pos_data['user_id'].values
    pos_item_ids = pos_data['item_id'].values

    neg_item_ids = rng.randint(low=1, high=max_item_id + 1,
                               size=len(user_ids))

    return [user_ids, pos_item_ids, neg_item_ids]


# Let's train the triplet model:

# In[ ]:


# we plug the identity loss and the a fake target variable ignored by
# the model to be able to use the Keras API to train the triplet model
triplet_model.compile(loss=identity_loss, optimizer="adam")
fake_y = np.ones_like(pos_data_train['user_id'])

n_epochs = 15

for i in range(n_epochs):
    # Sample new negatives to build different triplets at each epoch
    triplet_inputs = sample_triplets(pos_data_train, max_item_id,
                                     random_seed=i)

    # Fit the model incrementally by doing a single pass over the
    # sampled triplets.
    triplet_model.fit(triplet_inputs, fake_y, shuffle=True,
                      batch_size=64, epochs=1)
    
    # Monitor the convergence of the model
    test_auc = average_roc_auc(match_model, pos_data_train, pos_data_test)
    print("Epoch %d/%d: test ROC AUC: %0.4f"
          % (i + 1, n_epochs, test_auc))


# ## Training a Deep Matching Model on Implicit Feedback
# 
# 
# Instead of using hard-coded cosine similarities to predict the match of a `(user_id, item_id)` pair, we can instead specify a deep neural network based parametrisation of the similarity. The parameters of that matching model are also trained with the margin comparator loss:
# 
# <img src="images/rec_archi_implicit_1.svg" style="width: 600px;" />
# 
# 
# ### Exercise to complete as a home assignment:
# 
# - Implement a `deep_match_model`, `deep_triplet_model` pair of models
#   for the architecture described in the schema.   The last layer of
#   the embedded Multi Layer Perceptron outputs a single scalar that
#   encodes the similarity between a user and a candidate item.
# 
# - Evaluate the resulting model by computing the per-user average
#   ROC AUC score on the test feedback data.
#   
#   - Check that the AUC ROC score is close to 0.50 for a randomly
#     initialized model.
#     
#   - Check that you can reach at least 0.91 ROC AUC with this deep
#     model (you might need to adjust the hyperparameters).
#     
#     
# Hints:
# 
# - it is possible to reuse the code to create embeddings from the previous model
#   definition;
# 
# - the concatenation between user and the positive item embedding can be
#   obtained with:
# 
# ```py
#     positive_embeddings_pair = merge([user_embedding, positive_item_embedding],
#                                      mode='concat',
#                                      name="positive_embeddings_pair")
#     negative_embeddings_pair = merge([user_embedding, negative_item_embedding],
#                                      mode='concat',
#                                      name="negative_embeddings_pair")
# ```
# 
# - those embedding pairs should be fed to a shared MLP instance to compute the similarity scores.

# In[ ]:


from keras.models import Sequential


def make_interaction_mlp(input_dim, n_hidden=1, hidden_size=64,
                         dropout=0, l2_reg=None):
    mlp = Sequential()
    # TODO:
    return mlp


def build_models(n_users, n_items, user_dim=32, item_dim=64,
                 n_hidden=1, hidden_size=64, dropout=0, l2_reg=0):
    # TODO:
    # Inputs and the shared embeddings can be defined as previously.

    # Use a single instance of the MLP created by make_interaction_mlp()
    # and use it twice: once on the positive pair, once on the negative
    # pair
    interaction_layers = make_interaction_mlp(
        user_dim + item_dim, n_hidden=n_hidden, hidden_size=hidden_size,
        dropout=dropout, l2_reg=l2_reg)

    # Build the models: one for inference, one for triplet training
    deep_match_model = None
    deep_triplet_model = None
    return deep_match_model, deep_triplet_model


# In[ ]:


# %load solutions/deep_implicit_feedback_recsys.py


# ### Exercise:
# 
# Count the number of parameters in `deep_match_model` and `deep_triplet_model`. Which model has the largest number of parameters?

# In[ ]:


# print(deep_match_model.summary())


# In[ ]:


# print(deep_triplet_model.summary())


# In[ ]:


# %load solutions/deep_triplet_parameter_count.py


# ## Possible Extensions
# 
# You can implement any of the following ideas if you want to get a deeper understanding of recommender systems.
# 
# 
# ### Leverage User and Item metadata
# 
# As we did for the Explicit Feedback model, it's also possible to extend our models to take additional user and item metadata as side information when computing the match score.
# 
# 
# ### Better Ranking Metrics
# 
# In this notebook we evaluated the quality of the ranked recommendations using the ROC AUC metric. This score reflect the ability of the model to correctly rank any pair of items (sampled uniformly at random among all possible items).
# 
# In practice recommender systems will only display a few recommendations to the user (typically 1 to 10). It is typically more informative to use an evaluatio metric that characterize the quality of the top ranked items and attribute less or no importance to items that are not good recommendations for a specific users. Popular ranking metrics therefore include the **Precision at k** and the **Mean Average Precision**.
# 
# You can read up online about those metrics and try to implement them here.
# 
# 
# ### Hard Negatives Sampling
# 
# In this experiment we sampled negative items uniformly at random. However, after training the model for a while, it is possible that the vast majority of sampled negatives have a similarity already much lower than the positive pair and that the margin comparator loss sets the majority of the gradients to zero effectively wasting a lot of computation.
# 
# Given the current state of the recsys model we could sample harder negatives with a larger likelihood to train the model better closer to its decision boundary. This strategy is implemented in the WARP loss [1].
# 
# The main drawback of hard negative sampling is increasing the risk of sever overfitting if a significant fraction of the labels are noisy.
# 
# 
# ### Factorization Machines
# 
# A very popular recommender systems model is called Factorization Machines [2][3]. They two use low rank vector representations of the inputs but they do not use a cosine similarity or a neural network to model user/item compatibility.
# 
# It is be possible to adapt our previous code written with Keras to replace the cosine sims / MLP with the low rank FM quadratic interactions by reading through [this gentle introduction](http://tech.adroll.com/blog/data-science/2015/08/25/factorization-machines.html).
# 
# If you choose to do so, you can compare the quality of the predictions with those obtained by the [pywFM project](https://github.com/jfloff/pywFM) which provides a Python wrapper for the [official libFM C++ implementation](http://www.libfm.org/). Maciej Kula also maintains a [lighfm](http://www.libfm.org/) that implements an efficient and well documented variant in Cython and Python.
# 
# 
# ## References:
# 
#     [1] Wsabie: Scaling Up To Large Vocabulary Image Annotation
#     Jason Weston, Samy Bengio, Nicolas Usunier, 2011
#     https://research.google.com/pubs/pub37180.html
# 
#     [2] Factorization Machines, Steffen Rendle, 2010
#     https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf
# 
#     [3] Factorization Machines with libFM, Steffen Rendle, 2012
#     in ACM Trans. Intell. Syst. Technol., 3(3), May.
#     http://doi.acm.org/10.1145/2168752.2168771
