import torch
import numpy as np


class Metrics:
	def __init__(self):
		self.name = 'Metric Name'

	def reset(self):
		pass

	def update(self, predicts, batch):
		pass

	def get_score(self):
		pass


class Recall(Metrics):
	"""
	Args:
		 ats (int): @ to eval.
		 rank_na (bool): whether to consider no answer.
	"""
	def __init__(self, at=10):
		self.at = at
		self.n = 0
		self.n_correct = 0
		self.name = 'Recall@{}'.format(at)

	def reset(self):
		self.n = 0
		self.n_corrects = 0

	def update(self, predicts, batch):
		"""
		Args:
			predicts (FloatTensor): with size (batch, n_samples).
			batch (dict): batch.
		"""
		# TODO
		# This method will be called for each batch.
		# You need to
		# - increase self.n, which implies the total number of samples.
		# - increase self.n_corrects based on the prediction and labels
		#   of the batch.


		predicts = predicts.cpu()
		predicts = predicts.data.numpy()
		for i in range(predicts.shape[0]):
			self.n += 1
			
			sorted = np.argsort(predicts[i])[::-1]
			sorted = sorted[:self.at]
			for j in sorted:
				if batch['labels'][i][j] == 1:
					self.n_corrects += 1
					break
				

	def get_score(self):
		return self.n_corrects / self.n

	def print_score(self):
		score = self.get_score()
		return '{:.2f}'.format(score)
