
# coding: utf-8

# # Explicit Feedback Neural Recommender Systems
# 
# Goals:
# - Understand recommender data
# - Build different models architectures using Keras
# - Retrieve Embeddings and visualize them
# - Add metadata information as input to the model

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
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


# ### Ratings file
# 
# Each line contains a rated movie: 
# - a user
# - an item
# - a rating from 1 to 5 stars

# In[ ]:


import pandas as pd

raw_ratings = pd.read_csv(op.join(ML_100K_FOLDER, 'u.data'), sep='\t',
                      names=["user_id", "item_id", "rating", "timestamp"])
raw_ratings.head()


# ### Item metadata file
# 
# The item metadata file contains metadata like the name of the movie or the date it was released. The movies file contains columns indicating the movie's genres. Let's only load the first five columns of the file with `usecols`.

# In[ ]:


m_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
items = pd.read_csv(op.join(ML_100K_FOLDER, 'u.item'), sep='|',
                    names=m_cols, usecols=range(5), encoding='latin-1')
items.head()


# Let's write a bit of Python preprocessing code to extract the release year as an integer value:

# In[ ]:


def extract_year(release_date):
    if hasattr(release_date, 'split'):
        components = release_date.split('-')
        if len(components) == 3:
            return int(components[2])
    # Missing value marker
    return 1920


items['release_year'] = items['release_date'].map(extract_year)
items.hist('release_year', bins=50);


# Enrich the raw ratings data with the collected items metadata:

# In[ ]:


all_ratings = pd.merge(items, raw_ratings)


# In[ ]:


all_ratings.head()


# ### Data preprocessing
# 
# To understand well the distribution of the data, the following statistics are computed:
# - the number of users
# - the number of items
# - the rating distribution
# - the popularity of each movie

# In[ ]:


max_user_id = all_ratings['user_id'].max()
max_user_id


# In[ ]:


max_item_id = all_ratings['item_id'].max()
max_item_id


# In[ ]:


all_ratings['rating'].describe()


# Let's do a bit more pandas magic compute the popularity of each movie (number of ratings):

# In[ ]:


popularity = all_ratings.groupby('item_id').size().reset_index(name='popularity')
items = pd.merge(popularity, items)
items.nlargest(10, 'popularity')


# Enrich the ratings data with the popularity as an additional metadata.

# In[ ]:


all_ratings = pd.merge(popularity, all_ratings)
all_ratings.head()


# Later in the analysis we will assume that this popularity does not come from the ratings themselves but from an external metadata, e.g. box office numbers in the month after the release in movie theaters.
# 
# Let's split the enriched data in a train / test split to make it possible to do predictive modeling:

# In[ ]:


from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(
    all_ratings, test_size=0.2, random_state=0)

user_id_train = train_data['user_id']
item_id_train = train_data['item_id']
rating_train = train_data['rating']

user_id_test = test_data['user_id']
item_id_test = test_data['item_id']
rating_test = test_data['rating']


# # Explicit feedback: supervised ratings prediction
# 
# For each pair of (user, item) try to predict the rating the user would give to the item.
# 
# This is the classical setup for building recommender systems from offline data with explicit supervision signal. 

# ## Predictive ratings  as a regression problem
# 
# The following code implements the following architecture:
# 
# <img src="images/rec_archi_1.svg" style="width: 600px;" />

# In[ ]:


from tensorflow.contrib import keras
from keras.layers import Input, Embedding, Flatten, Dense, Dropout
from keras.layers import Dot
from keras.models import Model


# In[ ]:


# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 30
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

y = Dot(axes=1)([user_vecs, item_vecs])
# y = merge([user_vecs, item_vecs], mode=dot_mode, output_shape=(1,))

model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(optimizer='adam', loss='mae')


# In[ ]:


# Useful for debugging the output shape of model
initial_train_preds = model.predict([user_id_train, item_id_train])
initial_train_preds.shape


# ### Model error
# 
# Using `initial_train_preds`, compute the model errors:
# - mean absolute error
# - mean squared error
# 
# Converting a pandas Series to numpy array is usually implicit, but you may use `rating_train.values` to do so explicitly. Be sure to monitor the shapes of each object you deal with by using `object.shape`.

# In[ ]:


# %load solutions/compute_errors.py


# ### Monitoring runs
# 
# Keras enables to monitor various variables during training. 
# 
# `history.history` returned by the `model.fit` function is a dictionary
# containing the `'loss'` and validation loss `'val_loss'` after each epoch

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# Training the model\nhistory = model.fit([user_id_train, item_id_train], rating_train,\n                    batch_size=64, epochs=6, validation_split=0.1,\n                    shuffle=True)')


# In[ ]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('Loss');


# **Questions**:
# 
# - Why is the train loss higher than the first loss in the first few epochs?
# - Why is Keras not computing the train loss on the full training set at the end of each epoch as it does on the validation set?
# 
# 
# Now that the model is trained, the model MSE and MAE look nicer:

# In[ ]:


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))


# In[ ]:


train_preds = model.predict([user_id_train, item_id_train])
print("Final train MSE: %0.3f" % mean_squared_error(train_preds, rating_train))
print("Final train MAE: %0.3f" % mean_absolute_error(train_preds, rating_train))


# ## A Deep recommender model
# 
# Using a similar framework as previously, the following deep model described in the course was built (with only two fully connected)
# 
# <img src="images/rec_archi_2.svg" style="width: 600px;" />
# 
# To build this model we will need a new kind of layer:

# In[ ]:


from keras.layers import Concatenate


# 
# ### Exercise
# 
# - The following code has **4 errors** that prevent it from working correctly. **Correct them and explain** why they are critical.

# In[ ]:


# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')

embedding_size = 30
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)

# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = Concatenate()([user_vecs, item_vecs])
input_vecs = Dropout(0.99)(input_vecs)

x = Dense(64, activation='relu')(input_vecs)
y = Dense(2, activation='tanh')(x)

model = Model(inputs=[user_id_input, item_id_input], outputs=y)
model.compile(optimizer='adam', loss='binary_crossentropy')

initial_train_preds = model.predict([user_id_train, item_id_train])


# In[ ]:


# %load solutions/deep_explicit_feedback_recsys.py


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit([user_id_train, item_id_train], rating_train,\n                    batch_size=64, epochs=5, validation_split=0.1,\n                    shuffle=True)')


# In[ ]:


plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='validation')
plt.ylim(0, 2)
plt.legend(loc='best')
plt.title('Loss');


# In[ ]:


train_preds = model.predict([user_id_train, item_id_train])
print("Final train MSE: %0.3f" % mean_squared_error(train_preds, rating_train))
print("Final train MAE: %0.3f" % mean_absolute_error(train_preds, rating_train))


# In[ ]:


test_preds = model.predict([user_id_test, item_id_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))


# ### Home assignment: 
#  - Add another layer, compare train/test error
#  - What do you notice? 
#  - Try adding more dropout and modifying layer sizes: should you increase
#    or decrease the number of parameters

# ### Model Embeddings
# 
# - It is possible to retrieve the embeddings by simply using the Keras function `model.get_weights` which returns all the model learnable parameters.
# - The weights are returned the same order as they were build in the model
# - What is the total number of parameters?

# In[ ]:


# weights and shape
weights = model.get_weights()
[w.shape for w in weights]


# In[ ]:


# Solution: 
# model.summary()


# In[ ]:


user_embeddings = weights[0]
item_embeddings = weights[1]
print("First item name from metadata:", items["title"][1])
print("Embedding vector for the first item:")
print(item_embeddings[1])
print("shape:", item_embeddings[1].shape)


# ### Finding most similar items
# Finding k most similar items to a point in embedding space
# 
# - Write in numpy a function to compute the cosine similarity between two points in embedding space
# - Write a function which computes the euclidean distance between a point in embedding space and all other points
# - Write a most similar function, which returns the k item names with lowest euclidean distance
# - Try with a movie index, such as 181 (Return of the Jedi). What do you observe? Don't expect miracles on such a small training set.
# 
# Notes:
# - you may use `np.linalg.norm` to compute the norm of vector, and you may specify the `axis=`
# - the numpy function `np.argsort(...)` enables to compute the sorted indices of a vector
# - `items["name"][idxs]` returns the names of the items indexed by array idxs

# In[ ]:


EPSILON = 1e-07

def cosine(x, y):
    # TODO: modify function
    return 0.

# Computes euclidean distances between x and all item embeddings
def euclidean_distances(x):
    # TODO: modify function
    return 0.

# Computes top_n most similar items to an idx
def most_similar(idx, top_n=10):
    # TODO: modify function
    idxs = np.array([1,2,3])
    return items["title"][idxs]

most_similar(181)


# In[ ]:


# %load solutions/similarity.py


# ### Visualizing embeddings using TSNE
# 
# - we use scikit learn to visualize items embeddings
# - Try different perplexities, and visualize user embeddings as well
# - What can you conclude ?

# In[ ]:


from sklearn.manifold import TSNE

item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(item_tsne[:, 0], item_tsne[:, 1]);
plt.xticks(()); plt.yticks(());
plt.show()


# Alternatively with [Uniform Manifold Approximation and Projection](https://github.com/lmcinnes/umap):

# In[ ]:


# !pip install umap-learn


# In[ ]:


# import umap

# item_umap = umap.UMAP().fit_transform(item_embeddings)
# plt.figure(figsize=(10, 10))
# plt.scatter(item_umap[:, 0], item_umap[:, 1]);
# plt.xticks(()); plt.yticks(());
# plt.show()


# ## Using item metadata in the model
# 
# Using a similar framework as previously, we will build another deep model that can also leverage additional metadata. The resulting system is therefore an **Hybrid Recommender System** that does both **Collaborative Filtering** and **Content-based recommendations**.
# 
# <img src="images/rec_archi_3.svg" style="width: 600px;" />
# 

# In[ ]:


from sklearn.preprocessing import QuantileTransformer

meta_columns = ['popularity', 'release_year']

scaler = QuantileTransformer()
item_meta_train = scaler.fit_transform(train_data[meta_columns])
item_meta_test = scaler.transform(test_data[meta_columns])


# In[ ]:


# For each sample we input the integer identifiers
# of a single user and a single item
user_id_input = Input(shape=[1], name='user')
item_id_input = Input(shape=[1], name='item')
meta_input = Input(shape=[2], name='meta_item')

embedding_size = 32
user_embedding = Embedding(output_dim=embedding_size, input_dim=max_user_id + 1,
                           input_length=1, name='user_embedding')(user_id_input)
item_embedding = Embedding(output_dim=embedding_size, input_dim=max_item_id + 1,
                           input_length=1, name='item_embedding')(item_id_input)


# reshape from shape: (batch_size, input_length, embedding_size)
# to shape: (batch_size, input_length * embedding_size) which is
# equal to shape: (batch_size, embedding_size)
user_vecs = Flatten()(user_embedding)
item_vecs = Flatten()(item_embedding)

input_vecs = Concatenate()([user_vecs, item_vecs, meta_input])

x = Dense(64, activation='relu')(input_vecs)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
y = Dense(1)(x)

model = Model(inputs=[user_id_input, item_id_input, meta_input], outputs=y)
model.compile(optimizer='adam', loss='mae')

initial_train_preds = model.predict([user_id_train, item_id_train, item_meta_train])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'history = model.fit([user_id_train, item_id_train, item_meta_train], rating_train,\n                    batch_size=64, epochs=15, validation_split=0.1,\n                    shuffle=True)')


# In[ ]:


test_preds = model.predict([user_id_test, item_id_test, item_meta_test])
print("Final test MSE: %0.3f" % mean_squared_error(test_preds, rating_test))
print("Final test MAE: %0.3f" % mean_absolute_error(test_preds, rating_test))


# The additional metadata seem to improve the predictive power of the model a bit at least in terms of MAE.
# 
# 
# ### A recommendation function for a given user
# 
# Once the model is trained, the system can be used to recommend a few items for a user, that he/she hasn't already seen:
# - we use the `model.predict` to compute the ratings a user would have given to all items
# - we build a reco function that sorts these items and exclude those the user has already seen

# In[ ]:


indexed_items = items.set_index('item_id')


def recommend(user_id, top_n=10):
    item_ids = range(1, max_item_id)
    seen_mask = all_ratings["user_id"] == user_id
    seen_movies = set(all_ratings[seen_mask]["item_id"])
    item_ids = list(filter(lambda x: x not in seen_movies, item_ids))

    print("User %d has seen %d movies, including:" % (user_id, len(seen_movies)))
    for title in all_ratings[seen_mask].nlargest(20, 'popularity')['title']:
        print("   ", title)
    print("Computing ratings for %d other movies:" % len(item_ids))
    
    item_ids = np.array(item_ids)
    user_ids = np.zeros_like(item_ids)
    user_ids[:] = user_id
    items_meta = scaler.transform(indexed_items[meta_columns].loc[item_ids])
    
    rating_preds = model.predict([user_ids, item_ids, items_meta])
    
    item_ids = np.argsort(rating_preds[:, 0])[::-1].tolist()
    rec_items = item_ids[:top_n]
    return [(items["title"][movie], rating_preds[movie][0])
            for movie in rec_items]


# In[ ]:


for title, pred_rating in recommend(5):
    print("    %0.1f: %s" % (pred_rating, title))


# ### Home assignment: Predicting ratings as a classification problem
# 
# In this dataset, the ratings all belong to a finite set of possible values:

# In[ ]:


import numpy as np

np.unique(rating_train)


# Maybe we can help the model by forcing it to predict those values by treating the problem as a multiclassification problem. The only required changes are:
# 
# - setting the final layer to output class membership probabities using a softmax activation with 5 outputs;
# - optimize the categorical cross-entropy classification loss instead of a regression loss such as MSE or MAE.

# In[ ]:


# %load solutions/classification.py

