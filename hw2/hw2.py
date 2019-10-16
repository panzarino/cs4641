import numpy, matplotlib, os.path
from datetime import datetime
########################################
# FUNCTIONS YOU WILL NEED TO MODIFY:
#  - KNN.euclid
#  - KNN.manhattan
#  - KNN.mahalanobis
#  - KNN.query_single_pt
#  - LogisticRegression.prob_single_pt
#  - LogisticRegression.NLL_gradient
########################################

## Data loading utility functions
def get_test_train(fname,seed,datatype):
	'''
	Returns a test/train split of the data in fname shuffled with
	the given seed


	Args:
		fname: 		A str/file object that points to the CSV file to load, passed to
					numpy.genfromtxt()
		seed:		The seed passed to numpy.random.seed(). Typically an int or long
		datatype:	The datatype to pass to genfromtxt(), usually int, float, or str


	Returns:
		train_X:	A NxD numpy array of training data (row-vectors), 80% of all data
		train_Y:	A Nx1 numpy array of class labels for the training data
		test_X:		A MxD numpy array of testing data, same format as train_X, 20% of all data
		test_Y:		A Mx1 numpy array of class labels for the testing data
	'''
	data = numpy.genfromtxt(fname,delimiter=',',dtype=datatype)
	numpy.random.seed(seed)
	shuffled_idx = numpy.random.permutation(data.shape[0])
	cutoff = int(data.shape[0]*0.8)
	train_data = data[shuffled_idx[:cutoff]]
	test_data = data[shuffled_idx[cutoff:]]
	train_X = train_data[:,:-1].astype(float)
	train_Y = train_data[:,-1].reshape(-1,1)
	test_X = test_data[:,:-1].astype(float)
	test_Y = test_data[:,-1].reshape(-1,1)
	return train_X, train_Y, test_X, test_Y


def load_HTRU2(path='data'):
	return get_test_train(os.path.join(path,'HTRU_2.csv'),seed=1567708903,datatype=float)

def load_iris(path='data'):
	return get_test_train(os.path.join(path,'iris.data'),seed=1567708904,datatype=str)

## The "digits" dataset has a pre-split set of data, so we won't do our own test/train split
def load_digits(path='data'):
	train_data = numpy.genfromtxt(os.path.join(path,'optdigits.tra'),delimiter=',',dtype=float)
	test_data = numpy.genfromtxt(os.path.join(path,'optdigits.tes'),delimiter=',',dtype=float)
	return train_data[:,:-1], train_data[:,-1].reshape(-1,1), test_data[:,:-1], test_data[:,-1].reshape(-1,1)

## You can use this dataset to debug your implementation
def load_test(path='data'):
	return get_test_train(os.path.join(path,'data1.dat'),seed=1568572210, datatype=float)

class KNN():
	'''
	A very simple instance-based classifier.


	Finds the majority label of the k-nearest training points. Implements 3 distance functions:
	 - euclid: standard euclidean distance (l2)
	 - manhattan: taxi-cab, or l1 distance
	 - mahalanobis: euclidean distance after centering and scaling by the inverse of
	   the standard deviation of each component in the training data
	'''

	def __init__(self, train_X, train_Y):
		'''
		Stores the training data and computes some useful statistics for the mahalanobis
		distance method


		Args:
			train_X: NxD numpy array of training points. Should be row-vector form
			train_Y: Nx1 numpy array of class labels for training points.
		'''
		self.train_X = train_X
		self.train_Y = train_Y
		self.x_mean = train_X.mean(axis=0)
		self.x_std = train_X.std(axis=0)
		self.train_X_centered_scaled = (train_X-numpy.tile(self.x_mean,(train_X.shape[0],1)))/self.x_std

	def euclid(self,x_q):
		'''
		Returns a numpy array containing the euclidean distance from the test point, x_q
		to each of the points in the training data.


		Args:
			x_q:	The query point, a row vector with the same shape as one row of the
					training data in self.train_X


		Returns:
			dists:	A Nx1 numpy array containing the distance to each of the training
					data points
		'''
		# TODO: YOUR CODE HERE
		dists = numpy.zeros((self.train_X.shape[0],1))
		i = 0
		for x in self.train_X:
			diff = x - x_q
			dists[i] = numpy.sqrt(diff.T.dot(diff))
			i += 1
		return dists
	def manhattan(self,x_q):
		'''
		Returns a numpy array containing the manhattan distance from the test point, x_q
		to each of the points in the training data.


		Args:
			x_q:	The query point, a row vector with the same shape as one row of the
					training data in self.train_X


		Returns:
			dists:	A Nx1 numpy array containing the distance to each of the training
					data points
		'''
		# TODO: YOUR CODE HERE
		dists = numpy.zeros((self.train_X.shape[0],1))
		i = 0
		for x in self.train_X:
			dists[i] = numpy.sum(abs(x_q - x))
			i += 1
		return dists
	def mahalanobis(self,x_q):
		'''
		Returns a numpy array containing the centered and normalized distance from the test
		point, x_q to each of the points in the training data.


		Args:
			x_q:	The query point, a row vector with the same shape as one row of the
					training data in self.train_X


		Returns:
			dists:	A Nx1 numpy array containing the distance to each of the training
					data points
		'''
		# TODO: YOUR CODE HERE
		x_q_tilde = x_q # remember to center and scale the query point
		dists = numpy.zeros((self.train_X.shape[0], 1))
		sigma = numpy.diag(numpy.square(self.x_std.T))
		i = 0
		for x in self.train_X:
			diff = x_q_tilde - x
			dists[i] = diff.T.dot(sigma).dot(diff)
			i += 1
		return dists
	def query_single_pt(self,query_X,k,d):
		'''
		Returns the most common class label of the k-neighbors with the lowest distance as
		computed by the distance function d


		Args:
			query_X:	The query point, a row vector with the same shape as one row of the
						training data in self.train_X
			k:			The number of neighbors to check
			d:			The distance function to use


		Returns:
			label:		The label of the most common class
		'''
		## Note: the argument d is a function pointer. You could pass in knn.euclid or
		## knn.manhattan or knn.mahalanobis, or any other function you like, and here's
		## how you might use it
		# TODO: YOUR CODE HERE
		counts = numpy.unique(self.train_Y[numpy.argpartition(d(query_X).flatten(), k)[:k]].flatten(), return_counts=True)
		label = counts[0][numpy.argmax(counts[1])]
		return label
	def query(self,data_X,k,d):
		'''
		A convenience method for calling query_single_pt on each point in a dataset,
		such as those returned by get_test_train() or the various load_*() functions.
		If you change the API for any of the other methods, you'll probably have to rewrite
		this one.
		'''
		return numpy.array([self.query_single_pt(x_pt,k,d) for x_pt in data_X]).reshape(-1,1)
	def test_loss(self,max_k,d,test_X,test_Y):
		'''
		A convenience method for computing the misclasification rate for a range of k
		values.


		Args:
			max_k:	The maximum size of the neighborhood to test. Note, if you let this
					be too large, it may take a very long time to finish. You may want
					to add some code to print out after each iteration to see how long
					things are taking
			d:		The distance function to use
			test_X:	The data to compute the misclassification rate for
			test_Y:	The correct labels for the test data


		Returns:
			loss:	A numpy array with max_k entries containing the misclassification
					rate on the given data for each value of k. Useful for passing to
					matplotlib plotting methods.
		'''
		loss = numpy.zeros(max_k)
		i = 0
		for k in range(1, max_k + 1):
			start = datetime.now()
			loss[i] = (test_Y != self.query(test_X, k, d)).sum() / float(test_X.shape[0])
			i += 1
			print("Completed k={} time={}".format(k, datetime.now() - start))
		return loss
	def train_loss(self, max_k, d):
		'''
		A convenience method which calls self.test_loss() on the training data. Same
		arguments as for test_loss.
		'''
		return self.test_loss(max_k,d,self.train_X,self.train_Y)

## You should use this function for computing the sigmoid, as it prevents numerical
## overflow and underflow
def sigmoid(z):
	return numpy.clip(1.0/(1.0+numpy.exp(-z)),numpy.finfo(float).eps,1.0-numpy.finfo(float).eps)

class LogisticRegression():
	'''
	A simple linear classifier for binary datasets


	Finds the best linear decision boundary, and models the probability of the postive class
	label. Implements gradient descent with momentum to fit the model. Note that most
	of the methods provided to you in this class assume that data passed in (train_X, query_X,
	etc.) do not already include a constant offset term. You may change this if you like, but
	you will probably have to change some of the provided code to match.
	'''
	def __init__(self, train_X, train_Y, positive_class_label):
		'''
		Stores the training data, the class label for positive samples, and initializes
		self.Theta to None.


		Args:
			train_X:
				A NxD numpy array of training data points (row-vector format)

			train_Y:
				A Nx1 numpy array of class labels for each of the training points

			positive_class_label:
				The label of the "positive" class. Should be one of the labels in train_Y.
		'''
		self.train_X = train_X
		self.train_Y = train_Y
		self.positive_class_label = positive_class_label
		self.Theta = None

	def prob_single_pt(self,query_X):
		'''
		Returns the probability that query_X belongs to the positive class


		Args:
			query_X:	A numpy array with shape (D,)


		Returns:
			prob:		The probability that query_X belongs to the positive class
		'''
		#TODO: YOUR CODE HERE
		prob = sigmoid(numpy.hstack([numpy.ones(1), query_X]).dot(self.Theta))[0]
		return prob

	def prob(self,data_X):
		'''
		A convenience method for calling prob_single_pt for an array of test points


		Args:
			data_X:		A numpy array with shape (N,D)


		Returns:
			prob:		A numpy array with shape (N,1) with the probability of the class label
						for the corresponding point.
		'''
		return numpy.array([self.prob_single_pt(x_pt) for x_pt in data_X]).reshape(-1,1)

	def negative_log_likelihood(self,target_Theta=None):
		'''
		Computes the negative log likelihood of the data under the model with the given parameters


		Args:
			target_Theta:	the parameters to use, defaulting to self.Theta


		Returns:
			NLL:			The negative log likelihood.
		'''
		if target_Theta is None:
			target_Theta = self.Theta
		# Here's an example of turning train_Y into a vector of 0's and 1's
		# so that 1's match up with labels == positive_class_label and 0's
		# otherwise. You may find this useful in NLL_gradient
		Y = (self.train_Y == self.positive_class_label).astype(float)
		X = numpy.hstack([numpy.ones(self.train_X.shape[0]).reshape(-1,1),self.train_X])
		H = sigmoid(X.dot(target_Theta))
		NLL = -1.0*((Y*numpy.log(H)) + ((1.0-Y)*numpy.log(1.0-H))).sum()
		return NLL

	def NLL_gradient(self):
		'''
		Computes the gradient of the negative log likelihood with respect to the parameters
		self.Theta.


		Returns:
			grad:	A (D+1)x1 numpy array, column vector, representing the gradient
		'''
		# TODO: YOUR CODE HERE
		Y = (self.train_Y == self.positive_class_label).astype(float)
		X = numpy.hstack([numpy.ones(self.train_X.shape[0]).reshape(-1, 1), self.train_X])
		H = sigmoid(X.dot(self.Theta))
		grad = X.T.dot(H - Y)
		return grad

	def gradient_descent(self, initial_Theta, alpha=1e-3, num_iters=200000, print_every=20000, regularization=0.01, epsilon=1e-4):
		'''
		Performs gradient descent to optimize self.Theta to fit the training data


		Args:
			initial_Theta:	A numpy (D+1)x1 array to start the optimization from
			alpha:			The learning rate, a number
			num_iters:		The maximum number of iterations to perform, may stop before
			print_every:	Print out the current iteration, NLL, and Theta every so often
			regularization:	The factor to penalize ||Theta|| by. Necessary to prevent overfitting
							to linearly separable data
			epsilon:		Stop if ||Theta - prev_Theta|| < epsilon
		'''
		self.Theta = initial_Theta
		prev_Theta = self.Theta
		for k in range(1,num_iters+1):
			cur_NLL = self.negative_log_likelihood()
			if print_every and (k % print_every)==0:
				print("Iteration: {}, NLL: {}, Theta: {}".format(k,cur_NLL,self.Theta))
			# We're not including regularization in the gradient, and we shouldn't
			# regularize the constant offset term, so we add it here
			reg_term = self.Theta.copy()
			reg_term[0] = 0.0
			reg_term = regularization*reg_term
			grad = (self.NLL_gradient()+reg_term)
			# Note that we're scaling the gradient by the size of the training points. If you
			# already did this in the gradient function, you'll need to modify this line
			new_Theta = self.Theta - alpha*(grad/self.train_X.shape[0])
			prev_Theta = self.Theta
			self.Theta = new_Theta
			if numpy.linalg.norm(new_Theta-prev_Theta)<epsilon:
				print("Converged, iteration {}".format(k))
				break

def multiclass_logistic_score(list_of_logreg, test_X, test_Y):
	'''
	A utility function for computing the One-vs-Rest score of a list of logistic regression
	classifiers. O-v-R score compares the label of the highest-probability model with the
	true label. Higher is better. Score and misclassification loss are related by

	loss() = 1-score()

	make sure when you are comparing performance between KNN and LogisticRegression, you're
	comparing loss to loss, or score to score!


	Args:
		list_of_logreg:	A list of LogisticRegression objects, correctly initialized and fit
		test_X:			The data to test on
		test_Y:			The class labels for the testing data


	Returns:
		score:			The fraction of correctly identified labels.
	'''
	classes = [lr.positive_class_label for lr in list_of_logreg]
	probs = numpy.hstack([lr.prob(test_X) for lr in list_of_logreg])
	max_prob = numpy.argmax(probs,axis=1)
	Y_hat = numpy.array([classes[mp] for mp in max_prob]).reshape(-1,1)
	score = (Y_hat == test_Y).sum()/test_Y.shape[0]
	return score

def main():
	test_data = load_test()
	test_KNN = KNN(test_data[0],test_data[1])
	kNN_euc_loss = test_KNN.test_loss(5,test_KNN.euclid,test_data[2],test_data[3])
	kNN_man_loss = test_KNN.test_loss(5,test_KNN.manhattan,test_data[2],test_data[3])
	kNN_mah_loss = test_KNN.test_loss(5,test_KNN.mahalanobis,test_data[2],test_data[3])
	print("KNN loss:\n\t Euclid:{},(k={})\n\t Manhattan:{},(k={})\n\t Mahalanobis:{},(k={})".format(min(kNN_euc_loss),numpy.argmin(kNN_euc_loss)+1,
																									min(kNN_man_loss),numpy.argmin(kNN_man_loss)+1,
																									min(kNN_mah_loss),numpy.argmin(kNN_mah_loss)+1))
	test_logreg = LogisticRegression(test_data[0], test_data[1], 1)
	test_logreg.gradient_descent(initial_Theta=numpy.zeros((3,1)),epsilon=0.00005)
	lr_score = ((test_logreg.prob(test_data[2])>0.5) == (test_data[3]==1)).sum()/test_data[3].shape[0]
	print("LogReg score: {}".format(lr_score))

if __name__ == '__main__':
	main()