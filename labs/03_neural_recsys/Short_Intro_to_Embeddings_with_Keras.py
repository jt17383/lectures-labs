
# coding: utf-8

# # Categorical Embeddings
# 
# We will use the embeddings through the whole lab. They are simply represented by a matrix of tunable parameters (weights).
# 
# Let us assume that we are given a pre-trained embedding matrix for an vocabulary of size 10. Each embedding vector in that matrix has dimension 4. Those dimensions are too small to be realistic and are only used for demonstration purposes:

# In[ ]:


import numpy as np

embedding_size = 4
vocab_size = 10

embedding_matrix = np.arange(embedding_size * vocab_size, dtype='float32')
embedding_matrix = embedding_matrix.reshape(vocab_size, embedding_size)
print(embedding_matrix)


# To access the embedding for a given integer (ordinal) symbol $i$, you may either:
#  - simply index (slice) the embedding matrix by $i$, using numpy integer indexing:

# In[ ]:


i = 3
print(embedding_matrix[i])


#  - compute a one-hot encoding vector $\mathbf{v}$ of $i$, then compute a dot product with the embedding matrix:

# In[ ]:


def onehot_encode(dim, label):
    return np.eye(dim)[label]


onehot_i = onehot_encode(vocab_size, i)
print(onehot_i)


# In[ ]:


embedding_vector = np.dot(onehot_i, embedding_matrix)
print(embedding_vector)


# ### The Embedding layer in Keras
# 
# In Keras, embeddings have an extra parameter, `input_length` which is typically used when having a sequence of symbols as input (think sequence of words). In our case, the length will always be 1.
# 
# ```py
# Embedding(output_dim=embedding_size, input_dim=vocab_size,
#           input_length=sequence_length, name='my_embedding')
# ```
# 
# furthermore, we load the fixed weights from the previous matrix instead of using a random initialization:
# 
# ```py
# Embedding(output_dim=embedding_size, input_dim=vocab_size,
#           weights=[embedding_matrix],
#           input_length=sequence_length, name='my_embedding')
# ```

# In[ ]:


from keras.layers import Embedding

embedding_layer = Embedding(
    output_dim=embedding_size, input_dim=vocab_size,
    weights=[embedding_matrix],
    input_length=1, name='my_embedding')


# Let's use it as part of a Keras model:

# In[ ]:


from keras.layers import Input
from keras.models import Model

x = Input(shape=[1], name='input')
embedding = embedding_layer(x)
model = Model(inputs=x, outputs=embedding)


# The output of an embedding layer is then a 3-d tensor of shape `(batch_size, sequence_length, embedding_size)`.

# In[ ]:


model.output_shape


# `None` is a marker for dynamic dimensions.
# 
# The embedding weights can be retrieved as model parameters:

# In[ ]:


model.get_weights()


# The `model.summary()` method gives the list of trainable parameters per layer in the model:

# In[ ]:


model.summary()


# We can use the `predict` method of the Keras embedding model to project a single integer label into the matching embedding vector:

# In[ ]:


labels_to_encode = np.array([[3]])
model.predict(labels_to_encode)


# Let's do the same for a batch of integers:

# In[ ]:


labels_to_encode = np.array([[3], [3], [0], [9]])
model.predict(labels_to_encode)


# The output of an embedding layer is then a 3-d tensor of shape `(batch_size, sequence_length, embedding_size)`.
# To remove the sequence dimension, useless in our case, we use the `Flatten()` layer

# In[ ]:


from keras.layers import Flatten

x = Input(shape=[1], name='input')
y = Flatten()(embedding_layer(x))
model2 = Model(inputs=x, outputs=y)


# In[ ]:


model2.output_shape


# In[ ]:


model2.predict(np.array([3]))


# **Question** how many trainable parameters does `model2` have? Check your answer with `model2.summary()`.

# Note that we re-used the same `embedding_layer` instance in both `model` and `model2`: therefore **the two models share exactly the same weights in memory**:

# In[ ]:


model2.set_weights([np.ones(shape=(vocab_size, embedding_size))])


# In[ ]:


labels_to_encode = np.array([[3]])
model2.predict(labels_to_encode)


# In[ ]:


model.predict(labels_to_encode)


# **Home assignment**:
# 
# 
# The previous model definitions used the [function API of Keras](https://keras.io/getting-started/functional-api-guide/). Because the embedding and flatten layers are just stacked one after the other it is possible to instead use the [Sequential model API](https://keras.io/getting-started/sequential-model-guide/).
# 
# Defined a third model named `model3` using the sequential API and that also reuses the same embedding layer to share parameters with `model` and `model2`.

# In[ ]:


from keras.models import Sequential


# TODO
model3 = None

# print(model3.predict(labels_to_encode))


# In[ ]:


# %load solutions/embeddings_sequential_model.py

