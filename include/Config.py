import tensorflow as tf


class Config:
	dim = 300
	act_func = tf.nn.relu
	alpha = 0.1
	beta = 0.3
	gamma = 1.0  # margin based loss
	k = 125  # number of negative samples for each positive one
	seed = 3  # 30% of seeds

	def __init__(self, language='ja_en', epochs=600):
		self.language = language
		self.epochs = epochs
		self.e1 = 'data/' + language + '/ent_ids_1'
		self.e2 = 'data/' + language + '/ent_ids_2'
		self.ill = 'data/' + language + '/ref_ent_ids'
		self.kg1 = 'data/' + language + '/triples_1'
		self.kg2 = 'data/' + language + '/triples_2'
