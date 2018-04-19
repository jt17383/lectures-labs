
# coding: utf-8

# # Embeddings
# 
# We will use the embeddings through the whole lab. They are simply represented by a weight matrix:

# In[1]:


import numpy as np

embedding_size = 4
vocab_size = 10

embedding = np.arange(embedding_size * vocab_size, dtype='float')
embedding = embedding.reshape(vocab_size, embedding_size)
print(embedding)


# To access the embedding for a given symbol $i$, you may:
#  - compute a one-hot encoding of $i$, then compute a dot product with the embedding matrix
#  - simply index (slice) the embedding matrix by $i$, using numpy indexing

# In[2]:


i = 3
onehot = np.zeros(10)
onehot[i] = 1.
onehot


# In[3]:


embedding_vector = np.dot(onehot, embedding)
print(embedding_vector)


# In[4]:


print(embedding[i])


# ### The Embedding layer in Keras
# 
# In Keras, embeddings have an extra parameter, `input_length` which is typically used when having a sequence of symbols as input (think sequence of words). In our case, the length will always be 1.
# 
# ```py
# Embedding(output_dim=embedding_size, input_dim=vocab_size,
#           input_length=sequence_length, name='my_embedding')
# ```

# In[5]:


from keras.layers import Embedding

embedding_layer = Embedding(
    output_dim=embedding_size, input_dim=vocab_size,
    input_length=1, name='my_embedding')


# Let's use it as part of a Keras model:

# In[6]:


from keras.layers import Input
from keras.models import Model

x = Input(shape=[1], name='input')
embedding = embedding_layer(x)
model = Model(input=x, output=embedding)
model.output_shape


# The embedding weights are randomly initialized:

# In[7]:


model.get_weights()


# In[8]:


model.predict([[0],
               [3]])


# The output of an embedding layer is then a 3-d tensor of shape `(batch_size, sequence_length, embedding_size)`
# To remove the sequence dimension, useless in our case, we use the `Flatten()` layer

# In[9]:


from keras.layers import Flatten

x = Input(shape=[1], name='input')

# Add a flatten layer to remove useless "sequence" dimension
y = Flatten()(embedding_layer(x))

model2 = Model(input=x, output=y)
model2.output_shape


# In[10]:


model2.predict([[0],
                [3]])


# Note that we re-used the same `embedding_layer` instance in both `model` and `model2`: therefore the two model share exactly the same weights in memory.
