import numpy, matplotlib.pyplot, os.path
from mpl_toolkits.mplot3d import Axes3D
########################################
# FUNCTIONS YOU WILL NEED TO MODIFY:
# - PCA.find_components
# - PCA.transform
# - PCA.inv_transform
# - KMeans.cluster
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
def load_test2(path='data'):
	return get_test_train(os.path.join(path,'data2.dat'),seed=1568572211, datatype=float)

class PCA():
	'''
	A popular feature transformation/reduction/visualization method


	Uses the singular value decomposition to find the orthogonal directions of
	maximum variance.
	'''
	def __init__(self):
		'''
		Initializes some data members to hold the three components of the
		SVD.
		'''
		self.u = None
		self.s = None
		self.v = None
		self.shift = None
		self.data = None

	def find_components(self,data):
		'''
		Finds the SVD factorization and stores the result.


		Args:
			data: A NxD array of data points in row-vector format.
		'''
		self.data = data
		#NOTE: Make sure you center the data, as some of the provided
		#code expects this. If you don't center the data, you may
		#get different results

		self.shift = data.mean(axis=0)
		data = data - self.shift
		self.u, self.s, self.v = numpy.linalg.svd(data)


	def transform(self,n_components,data=None):
		'''
		Uses the values computed and stored after calling find_components()
		to transform the data into n_components dimensions.


		Args:
			n_components: The number of dimensions to transform the data into.
			data: 	the data to apply the transform to. Defaults to the data
					provided on the last call to find_components()


		Returns:
			transformed_data: 	a Nx(n_components) array of transformed points,
								in row-vector format.
		'''
		if data is None:
			data = self.data
		#NOTE: Don't forget to center the data

		data = data - self.shift
		return data.dot(self.v.T[:, :n_components])

	def inv_transform(self,n_components,transformed_data):
		'''
		Inverts the results of transform() (if given the same arguments).


		Args:
			n_components:		Number of components to use. Should match
								the dimension of transformed_data.
			transformed_data:	The data to apply the inverse transform to,
								should be in row-vector format


		Returns:
			inv_tform_data:		a NxD array of points in row-vector format
		'''
		#NOTE: Don't forget to "un-center" the data

		return transformed_data.dot(self.v[:, :n_components]) + self.shift

	def reconstruct(self,n_components,data=None):
		'''
		Casts the data down to n_components dimensions, and then reverses the transform,
		returning the low-rank approximation of the given data. Defaults to the data
		provided on the last call to find_components().
		'''
		return self.inv_transform(n_components,self.transform(n_components,data))

	def reconstruction_error(self,n_components,data=None):
		'''
		Useful for determining how much information is preserved in n_components dimensions.
		'''
		if data is None:
			data = self.data
		return numpy.linalg.norm(data-self.reconstruct(n_components,data),ord='fro')

	def plot_2D_proj(self,data=None,labels=None):
		'''
		Creates a 2D visualization of the data, returning the created figure. See
		the main() function for example usage.
		'''
		fig = matplotlib.pyplot.figure()
		proj_2d_data = self.transform(2,data)
		fig.gca().scatter(proj_2d_data[:,0],proj_2d_data[:,1],c=labels)
		fig.gca().set_title('PCA 2D transformation')
		return fig
	def plot_3D_proj(self,data=None,labels=None):
		'''
		Creates a 3D visualization of the data, returning the created figure. See
		the main() function for example usage.
		'''
		fig = matplotlib.pyplot.figure()
		ax = fig.add_subplot(111,projection='3d')
		proj_3d_data = self.transform(3,data)
		ax.scatter(proj_3d_data[:,0],proj_3d_data[:,1],proj_3d_data[:,2],c=labels)
		fig.gca().set_title('PCA 3D transformation')
		return fig



# You may find this method useful for providing a canonical ordering of cluster centers in KMeans.cluster
def consistent_ordering(array):
	rowvec_dists = numpy.linalg.norm(array,axis=1)
	dist_order = numpy.argsort(rowvec_dists,kind='stable')
	return array[dist_order,:]
class KMeans():
	'''
	A simple iterative clustering method for real-valued feature spaces.


	Finds cluster centers by iteratively assigning points to clusters and re-computing 
	cluster center locations. Provided code expects self.clusters to contain the
	cluster centers in row-vector format. 
	'''
	def __init__(self):
		'''
		Initializes data members.
		'''
		self.clusters = None
		self.print_every = 1000

	def cluster_distances(self,data):
		'''
		Computes the distance from each row of data to the cluster centers. Must call KMeans.cluster(), or
		otherwise set self.clusters first.


		Args:
			data:	The data to compute distances for, in row-vector format. Each row should have the
					same number of columns as each cluster


		Returns:
			dists:	A Nx(len(clusters)) array, one row for each row in data, one column for each
					cluster center, containing the distance for each point to each cluster
		'''
		return numpy.hstack([numpy.linalg.norm(data-c,axis=1).reshape(-1,1) for c in self.clusters])

	def cluster_label(self,data):
		'''
		Returns the label of the closest cluster to each row in data. Note that these labels are
		arbitrary, and do *not* correspond directly with class labels.


		Args:
			data:	Data to compute cluster labels for, in row-vector format.


		Returns:
			c_labels:	A N-by-1 array, one row for each row in data, containing the integer 
						corresponding to the cluster closest to each point.
		'''
		return numpy.argmin(self.cluster_distances(data),axis=1)

	def cluster(self,data,k):
		'''
		Implements the k-Means iterative algorithm. Cluster centers are initially chosen randomly,
		then on each iteration each data point is assigned to the closest cluster center, and then the
		cluster centers are re-computed by averaging the points assigned to them. After finishing, 
		self.clusters should contain a k-by-D array of the cluster centers in row-vector format.


		Args:
			data:	Data to be clustered in row-vector format. A N-by-D array.
			k:		The number of clusters to find.
		'''
		#This line should pick the initial clusters at random from the provided
		#data.
		self.clusters = data[numpy.random.choice(data.shape[0],k,replace=False),:]
		#An example of how to use the consistent_ordering() function, which you
		#may want to use to help determine if cluster centers have changed from one
		#iteration to the next
		self.clusters = consistent_ordering(self.clusters)
		#We know that k-Means will always converge, but depending on the initial
		#conditions and dataset, it may take a long time. For debugging purposes
		#you might want to set a maximum number of iterations. When implemented
		#correctly, none of the provided datasets take many iterations to converge
		#for most initial configurations.
		not_done = True
		itr = 0
		while not_done:
			itr += 1
			new_clusters = numpy.zeros(self.clusters.shape)
			changed = False
			# TODO: Your code here
			#assign points to nearest cluster
			#re-compute cluster centers by averaging the points assigned to them
			#determine if clusters have changed from the previous iteration
			#For debugging, print out every so often.

			closest_cluster = self.cluster_label(data)
			for i in range(len(self.clusters)):
				new_clusters[i] = data[numpy.where(closest_cluster == i)].mean(axis=0)
				if not changed and not numpy.array_equal(self.clusters[i], new_clusters[i]):
					changed = True
			not_done = changed

			if itr % self.print_every == 0:
				print("Iteration {}, change {}".format(itr,numpy.linalg.norm(new_clusters-self.clusters,ord='fro')))
			self.clusters = new_clusters
		print("Converged after {} iterations".format(itr))

	def normalized_mutual_information(self, data, labels):
		'''
		Since cluster assignments are not the same as class labels, we can't always directly
		compare them to measure clustering performance. However, we can measure the Mutual Information
		between two labelings, to see if they contain the same statistical information. This method
		implements the "Normalized Mutual Information Score" as described here:

		https://scikit-learn.org/stable/modules/clustering.html#mutual-information-based-scores

		Note that this version uses arithmetic mean, when comparing output with 
		sklearn.metrics.mutual_info_score()
		'''
		cluster_labels = self.cluster_label(data)
		P_cl = numpy.zeros(len(self.clusters))
		P_gt = numpy.zeros(len(numpy.unique(labels)))
		P_clgt = numpy.zeros((len(P_cl),len(P_gt)))
		cl_masks = dict()
		gt_masks = dict()
		MI = 0.0
		H_cl = 0.0
		H_gt = 0.0
		for c_cl in range(len(P_cl)):
			cl_masks[c_cl] = (cluster_labels==c_cl).reshape(-1,1)
			P_cl[c_cl] = (cl_masks[c_cl]).astype(int).sum()/len(data)
			H_cl -= P_cl[c_cl]*numpy.log(P_cl[c_cl])
		for c_gt in range(len(P_gt)):
			gt_masks[c_gt] = labels == c_gt
			P_gt[c_gt] = (gt_masks[c_gt]).astype(int).sum()/len(data)
			H_gt -= P_gt[c_gt]*numpy.log(P_gt[c_gt])
		for c_cl in range(len(P_cl)):
			for c_gt in range(len(P_gt)):
				P_clgt[c_cl,c_gt] = (numpy.logical_and(cl_masks[c_cl], gt_masks[c_gt])).astype(int).sum()/len(data)
				if P_clgt[c_cl,c_gt] == 0.0:
					MI += 0
				else:
					MI += P_clgt[c_cl,c_gt]*numpy.log(P_clgt[c_cl,c_gt]/(P_cl[c_cl]*P_gt[c_gt]))
		return MI/(numpy.mean([H_cl,H_gt]))

	def plot_2D_clusterd(self, data, labels=None):
		'''
		Creates a 2D visualization of the data, returning the created figure. See
		the main() function for example usage.
		'''		
		if self.clusters is None or len(self.clusters)!=2:
			self.cluster(data,2)
		fig = matplotlib.pyplot.figure()
		clusterd_2d_data = self.cluster_distances(data)
		fig.gca().scatter(clusterd_2d_data[:,0],clusterd_2d_data[:,1],c=labels)
		fig.gca().set_title('k-Means 2D cluster distance')
		return fig
	def plot_3D_clusterd(self, data, labels=None):
		'''
		Creates a 3D visualization of the data, returning the created figure. See
		the main() function for example usage.
		'''
		if self.clusters is None or len(self.clusters)!=3:
			self.cluster(data,3)
		fig = matplotlib.pyplot.figure()
		ax = fig.add_subplot(111,projection='3d')
		clusterd_3d_data = self.cluster_distances(data)
		ax.scatter(clusterd_3d_data[:,0],clusterd_3d_data[:,1],clusterd_3d_data[:,2],c=labels)
		fig.gca().set_title('k-Means 3D cluster distance')
		return fig

def main():
	data = load_test2()
	#### PCA
	pca = PCA()
	pca.find_components(data[0])
	# without labels
	# pca.plot_2D_proj(data[0])
	# with labels
	_, labels = numpy.unique(data[1], return_inverse=True)
	pca.plot_2D_proj(data[0], labels)
	# pca.plot_3D_proj(data[0],labels)
	####
	#### k-Means
	km = KMeans()
	# without labels
	# km.plot_2D_clusterd(data[0])
	# with labels
	km.plot_2D_clusterd(data[0], labels)
	# km.plot_3D_clusterd(data[0],labels)
	matplotlib.pyplot.show()
	print(km.normalized_mutual_information(data[0], data[1]))


if __name__ == '__main__':
	main()