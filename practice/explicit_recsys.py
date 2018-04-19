# coding: utf-8

# In[3]:


import numpy as np
import os
import logging
from zipfile import ZipFile
from urllib.request import urlretrieve
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import keras


EPSILON = 1e-07
DATA_FOLDER = '../data/'
RESULTS_FOLDER = '../results/'
ML_100K_URL = "http://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML_100K_FILENAME = os.path.join(DATA_FOLDER, ML_100K_URL.rsplit('/', 1)[1])
ML_100K_FOLDER = os.path.join(DATA_FOLDER, 'ml-100k')

metrics = []

def load_dataset():
	global raw_ratings, items
	if not os.path.exists(ML_100K_FILENAME):
		logging.info(f'downloading {ML_100K_URL}...')
		urlretrieve(ML_100K_URL, ML_100K_FILENAME)
	if not os.path.exists(ML_100K_FOLDER):
		logging.info(f'extracting {ML_100K_FILENAME} to {ML_100K_FOLDER}')
		ZipFile(ML_100K_FILENAME).extractall('.')
	raw_ratings = pd.read_csv(
		os.path.join(ML_100K_FOLDER, 'u.data'),
		sep='\t',
		names=['user_id', 'item_id', 'rating', 'timestamp']
	)
	items = pd.read_csv(
		os.path.join(ML_100K_FOLDER, 'u.item'),
		sep='|',
		names=['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'],
		usecols=range(5),
		encoding='latin-1'
	)


def extract_year(release_date):
	if hasattr(release_date, 'split'):
		components = release_date.split('-')
		if len(components) == 3:
			return int(components[2])
	return 1920


def show_release_year_hist():
	global items
	import matplotlib.pyplot as plt
	items.hist('release_year', bins=50)
	print('whatssss@@')


def preprocess_dataset():
	global max_user_id, max_item_id, all_ratings, items
	items['release_year'] = items['release_date'].map(extract_year)
	all_ratings = pd.merge(items, raw_ratings)
	max_user_id = all_ratings['user_id'].max()
	max_item_id = all_ratings['item_id'].max()
	popularity = all_ratings.groupby('item_id').size().reset_index(name='popularity')
	items = pd.merge(popularity, items)
	items.nlargest(10, 'popularity')
	all_ratings = pd.merge(popularity, all_ratings)


def split_dataset():
	global user_id_train, item_id_train, train_data, user_id_test, item_id_test, test_data, rating_train, rating_test
	train_data, test_data = train_test_split(all_ratings, test_size=0.2, random_state=0)
	user_id_train = train_data['user_id']
	item_id_train = train_data['item_id']
	rating_train = train_data['rating']
	user_id_test = test_data['user_id']
	item_id_test = test_data['item_id']
	rating_test = test_data['rating']


def calc_metrics(y_true, y_pred):
	mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
	mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
	return mae, mse


def fit_predict(model):
	return fit_predict_inputs(
		model,
		[user_id_train, item_id_train],
		[user_id_test, item_id_test])


def fit_predict_inputs(model, x_train, x_test):
	initial_train_preds = model.predict(x_train)
	logging.info(f'output shape: {initial_train_preds.shape} ')
	history = model.fit(
		x=x_train,
		y=rating_train,
		batch_size=64,
		epochs=6,
		validation_split=0.1,
		shuffle=True
	)
	train_preds = model.predict(x_train)
	test_preds = model.predict(x_test)
	mae, mse = calc_metrics(rating_train, train_preds)
	metrics.append([model.name, 'train', mae, mse])
	mae, mse = calc_metrics(rating_test, test_preds)
	metrics.append([model.name, 'test', mae, mse])
	return history, metrics


def rs_v1():
	# # explicit feedback: supervised ratings prediction
	user_id_input = keras.layers.Input(shape=[1], name='user')
	item_id_input = keras.layers.Input(shape=[1], name='item')
	embedding_size = 30
	user_embedding = keras.layers.Embedding(
		output_dim=embedding_size,
		input_dim=max_user_id + 1,
		input_length=1,
		name='user_embedding'
	)(user_id_input)
	item_embedding = keras.layers.Embedding(
		output_dim=embedding_size,
		input_dim=max_item_id + 1,
		input_length=1,
		name='item_embedding'
	)(item_id_input)
	user_vecs = keras.layers.Flatten()(user_embedding)
	item_vecs = keras.layers.Flatten()(item_embedding)
	y = keras.layers.Dot(axes=1)([user_vecs, item_vecs])
	model = keras.models.Model(inputs=[user_id_input, item_id_input], outputs=y)
	model.compile(optimizer='adam', loss='mae')
	return fit_predict(model)


def rs_v2():
	user_id_input = keras.layers.Input(shape=[1], name='user')
	item_id_input = keras.layers.Input(shape=[1], name='item')
	embedding_size = 30
	user_embedding = keras.layers.Embedding(
		output_dim=embedding_size,
		input_dim=max_user_id + 1,
		input_length=1,
		name='user_embedding'
	)(user_id_input)
	item_embedding = keras.layers.Embedding(
		output_dim=embedding_size,
		input_dim=max_item_id + 1,
		input_length=1,
		name='item_embedding'
	)(item_id_input)
	user_vecs = keras.layers.Flatten()(user_embedding)
	item_vecs = keras.layers.Flatten()(item_embedding)
	input_vecs = keras.layers.Concatenate()([user_vecs, item_vecs])
	input_vecs = keras.layers.Dropout(0.5)(input_vecs)
	x = keras.layers.Dense(units=64, activation='relu')(input_vecs)
	y = keras.layers.Dense(units=1)(x)
	model = keras.models.Model(inputs=[user_id_input, item_id_input], outputs=y)
	model.compile(optimizer='adam', loss='mae')
	return fit_predict(model)


def rs_v3():
	user_id_input = keras.layers.Input(shape=[1], name='user')
	item_id_input = keras.layers.Input(shape=[1], name='item')
	embedding_size=30
	user_embedding = keras.layers.Embedding(
		input_dim=max_user_id+1,
		input_length= 1,
		output_dim=embedding_size,
		name = 'user_embedding'
	)(user_id_input)
	item_embedding = keras.layers.Embedding(
		input_dim = max_item_id+1,
		input_length=1,
		output_dim=embedding_size,
		name='item_embedding'
	)(item_id_input)
	user_vecs = keras.layers.Flatten()(user_embedding)
	item_vecs = keras.layers.Flatten()(item_embedding)
	x = keras.layers.Concatenate()([user_vecs, item_vecs])
	#x = keras.layers.Flatten()(x)
	x = keras.layers.Dropout(0.5)(x)
	x = keras.layers.Dense(units=32, activation='relu')(x)
	x = keras.layers.Dropout(0.5)(x)
	x = keras.layers.Dense(units=16, activation='relu')(x)
	#x = keras.layers.Dropout(0.5)(x)
	y = keras.layers.Dense(units=1)(x)
	model = keras.models.Model(
		inputs = [user_id_input, item_id_input],
		outputs= y
	)
	model.compile(optimizer='adam', loss='mae')
	history, metrics= fit_predict(model)
	return model, history,metrics


def train_models():
	global rs1_hist, rs2_hist, model, rs3_hist, rs4_hist
	output_file = os.path.join(RESULTS_FOLDER, 'rs1_hist.csv')
	if not os.path.exists(output_file):
		rs1_hist, rs1_metrics = rs_v1()
		rs1_hist = pd.DataFrame(data=rs1_hist.history)
		rs1_hist.to_csv(output_file, index=False)
	else:
		rs1_hist = pd.read_csv(output_file)
	output_file = os.path.join(RESULTS_FOLDER, 'rs2_hist.csv')
	if not os.path.exists(output_file):
		rs2_hist, rs2_metrics = rs_v2()
		rs2_hist = pd.DataFrame(data=rs2_hist.history)
		rs2_hist.to_csv(output_file, index=False)
	else:
		rs2_hist = pd.read_csv(output_file)
	output_file = os.path.join(RESULTS_FOLDER, 'rs3_hist.csv')
	if not os.path.exists(output_file):
		model, rs3_hist, rs3_metrics = rs_v3()
		rs3_hist = pd.DataFrame(data=rs3_hist.history)
		rs3_hist.to_csv(output_file, index=False)
	else:
		rs3_hist = pd.read_csv(output_file)


	from sklearn.preprocessing import QuantileTransformer

	meta_columns = ['popularity', 'release_year']
	scaler = QuantileTransformer()
	item_meta_train = scaler.fit_transform(train_data[meta_columns])
	item_meta_test = scaler.transform(test_data[meta_columns])

	user_id_input = keras.layers.Input(shape=[1],name='user_input')
	item_id_input = keras.layers.Input(shape=[1],name='item_input')
	item_meta_input = keras.layers.Input(shape=[2],name='item_meta_input')

	embedding_size = 30
	user_embeddings = keras.layers.Embedding(
		input_dim=max_user_id + 1,
		input_length=1,
		output_dim=embedding_size,
		name='user_embedding'
	)(user_id_input)
	item_embeddings = keras.layers.Embedding(
		input_dim=max_item_id+1,
		input_length=1,
		output_dim=embedding_size,
		name='item_embedding'
	)(item_id_input)

	user_vecs = keras.layers.Flatten()(user_embeddings)
	item_vecs = keras.layers.Flatten()(item_embeddings)

	input_vecs = keras.layers.Concatenate()([user_vecs, item_vecs, item_meta_input])
	x = keras.layers.Dense(units=64, activation='relu')(input_vecs)
	x = keras.layers.Dropout(0.5)(x)
	x = keras.layers.Dense(units=32, activation='relu')(x)
	x = keras.layers.Dropout(0.5)(x)
	y = keras.layers.Dense(units=1)(x)

	model = keras.models.Model(
		inputs=[user_id_input, item_id_input, item_meta_input ],
		outputs=y
	)
	model.compile(optimizer='adam', loss='mae')

	x_train = [user_id_train, item_id_train, item_meta_train]
	x_test = [user_id_test, item_id_test, item_meta_test]
	rs4_hist, metrics = fit_predict_inputs(model,  x_train,x_test)
	rs4_hist = pd.DataFrame(data=rs4_hist.history)



def cosine(x,y):
	dot_pdt = np.dot(x, y.T)
	norms = np.linalg.norm(x) * np.linalg.norm(y)
	sim= dot_pdt / (norms + EPSILON)
	#sim2 = cosine_similarity(x.reshape(1,-1),y.reshape(1,-1))
	return sim


def cosine_similarities(x, item_embeddings):
	dot_pdts = np.dot(item_embeddings, x)
	norms = np.linalg.norm(x) * np.linalg.norm(item_embeddings, axis=1)
	return dot_pdts / (norms+EPSILON)


def euclidean_distances(x, item_embeddings):
	return np.linalg.norm(item_embeddings-x, axis=1)


def most_similar(idx, item_embeddings, items, top_n=10, mode='euclidean'):
	sorted_indexes=0
	if mode=='euclidean':
		logging.info(f'euclidean mode')
		dists = euclidean_distances(item_embeddings[idx], item_embeddings)
		sorted_indexes = np.argsort(dists)
		idxs = sorted_indexes[0:top_n]
		return list(zip(items['title'][idxs], dists[idxs]))
	else:
		logging.info(f'cosine mode')
		sims = cosine_similarities(item_embeddings[idx], item_embeddings)
		sorted_indexes = np.argsort(sims)[::-1]
		idxs = sorted_indexes[0:top_n]
		return list(zip(items['title'][idxs], sims[idxs]))


def analyze_metrics():
	pass
	# metrics = rs1_metrics
	# metrics.extend(rs2_metrics)
	# metrics = pd.DataFrame(data=metrics, columns=['method','split', 'mae','mse'])
	# metrics


def analyze_similarity():
	# similarity
	weights = model.get_weights()
	[w.shape for w in weights]
	model.summary()
	user_embeddings = weights[0]
	item_embeddings = weights[1]
	sample_title = items['title'][1]
	sample_embedding = item_embeddings[1]
	logging.info(f'title {sample_title}: {sample_embedding.shape}: {sample_embedding}')
	cos_sim = cosine(item_embeddings[1], item_embeddings[1])
	logging.info(f'cosine similarity betweeen item1 {cos_sim}')
	euc_dists = euclidean_distances(item_embeddings[1], item_embeddings)
	euc_dists[0:5]
	sample_ix = 181
	sample_title = items['title'][sample_ix]
	logging.info(f'items closest to {sample_title}')
	for title, dist in most_similar(sample_ix, item_embeddings, items, mode='euclidean'):
		logging.info(f'{title}: {dist}')


def visualize_embeddings():
	from sklearn.manifold import TSNE
	weights = model.get_weights()
	item_embeddings = weights[1]  # lookup with model.summary()
	item_tsne = TSNE(perplexity=30).fit_transform(item_embeddings)
	import matplotlib.pyplot as plt
	plt.figure(figsize=(10, 10))
	plt.scatter(item_tsne[:, 0], item_tsne[:, 1])
	plt.xticks(())
	plt.yticks(())
	plt.show()


def visualize_loss():
	import matplotlib.pyplot as plt
	plt.plot(rs1_hist.loss, label='train_simple')
	plt.plot(rs1_hist.val_loss, label='validation_simple')
	plt.plot(rs2_hist.loss, label='train_deep')
	plt.plot(rs2_hist.val_loss, label='validation_deep')
	plt.plot(rs3_hist.loss, label='train_deep_v2')
	plt.plot(rs3_hist.val_loss, label='validation_deep_v2')
	plt.plot(rs4_hist.loss, label='train_deep_v3')
	plt.plot(rs4_hist.val_loss, label='validation_deep_v3')
	plt.legend(loc='best')
	plt.title('loss')


if __name__=='__main__':
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
	logging.info('initializing...')
	load_dataset()
	preprocess_dataset()
	split_dataset()
	train_models()
	visualize_loss()
	analyze_metrics()
	analyze_similarity()
	visualize_embeddings()
	logging.info('done!')

