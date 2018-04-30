from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from scipy import stats, special
from copy import deepcopy


def generate_mvn_data(means, covariances, samples):
	if not len(means) == len(covariances) == len(samples):
		print "Dimensions do not align."
		exit()

	data_points = []
	for i in xrange(len(means)):
		data_points.append(np.random.multivariate_normal(means[i], covariances[i], size=(samples[i])))
	
	return data_points


def plot_data(data):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	colors = ['red', 'blue', 'purple']
	shapes = ['o', '^', '*']
	for i, cluster in enumerate(data):
		cluster = np.asarray(cluster)
		ax.scatter(cluster[:, 0], cluster[:, 1], cluster[:, 2], color=colors[i], marker=shapes[i])

	ax.set_xlabel('dimension 1')
	ax.set_ylabel('dimension 2')
	ax.set_zlabel('dimension 3')

	plt.legend(['Cluster 1', 'Cluster 2', 'Cluster 3'])
	plt.show()


def kmeans(data):
	points = []
	for c in data:
		points.extend(c)

	km = cluster.KMeans(n_clusters=len(data)-1).fit(points)

	# Compute the furthest 20 points and classify as deviant/outlier
	distance_to_nearest_centroid = []
	for point, label in zip(points, km.labels_):
		dist = np.linalg.norm(point - km.cluster_centers_[label])
		distance_to_nearest_centroid.append(dist)
	
	farthest = np.argpartition(distance_to_nearest_centroid, -20)[-20:]
	# Give the 20 most outliers a new label of cluster '2'
	for index in farthest:
		km.labels_[index] = 2

	# Hack to make our cluster 0 line up with kmeans
	if abs(km.cluster_centers_[0][0] - 3.5) > 1:
		km.cluster_centers_[0], km.cluster_centers_[1] = km.cluster_centers_[1], km.cluster_centers_[0]
		for index, label in enumerate(km.labels_):
			if label == 0:
				km.labels_[index] = 1
			elif label == 1:
				km.labels_[index] = 0

	return points, km


def gibbs(points, init_labels, k=3):

	prior_mean = np.mean(points, axis=0)
	prior_cov = np.cov(points, rowvar=False)

	print "Prior Information"
	print "-----------------"
	print "prior mean: ", prior_mean
	print "\n"
	print "prior covariance: ", prior_cov
	print "\n"
	print "prior probability: ", (1.0/3.0)


	# Set priors
	degrees_of_freedom = 2
	labels = deepcopy(init_labels)
	sampled_prior_cov = [prior_cov] * 3
	sampled_prior_mean = [prior_mean] * 3
	sampled_probability_vector = [1.0/3] * 3
	alpha = [5] * 3

	clusters = [[], [], []]
	for point, label in zip(points, labels):
		clusters[label].append(point)

	probability_guesses = []
	mean_guesses = []
	covariance_guesses = []

	# Begin sampling
	iterations = 3200
	for i in xrange(iterations):
		if i % 10 == 0:
			print i
		# Step 1 - simulate Covariance matrices from inverse Wishart
		for j in xrange(k):
			
			sum_of_squares = np.zeros(shape=(prior_cov.shape))
			for p in clusters[j]:
				sum_of_squares += np.outer(np.asarray(p-np.mean(clusters[j], axis=0)),
										   np.asarray(p-np.mean(clusters[j], axis=0)).T)

			dev_from_prior = (len(clusters[j])/len(clusters[j])+1) * \
							 (np.outer(np.asarray(np.mean(clusters[j], axis=0) - sampled_prior_mean[j]),
							 		   np.asarray(np.mean(clusters[j], axis=0) - sampled_prior_mean[j]).T))

			new_cov = stats.invwishart.rvs(df=degrees_of_freedom + len(clusters[j]),
										   scale=sampled_prior_cov[j] + sum_of_squares + dev_from_prior)
			sampled_prior_cov[j] = new_cov

		# Step 2 - simulate mean from mvn 
		for j in xrange(k):
			cluster_mean = (sampled_prior_mean[j] + (len(clusters[j])*np.mean(clusters[j], axis=0))) \
						   / (len(clusters[j]) + 1)
			sampled_prior_mean[j] = np.random.multivariate_normal(cluster_mean,
																  sampled_prior_cov[j]/(len(clusters[j])+1))
		
		# Step 3 - simulate probability vector from dirichlet
		sampled_probability_vector = stats.dirichlet.rvs([alpha[0]+len(clusters[0]),
										   	              alpha[1]+len(clusters[1]),
										                  alpha[2]+len(clusters[2])])

		# Step 4 - assign new classification variables v_i
		for ind, p in enumerate(points):
			a = labels[ind]
			cluster_probabilities = []
			for j in xrange(k):
				prob = stats.multivariate_normal.pdf(p,
													 mean=sampled_prior_mean[j],
													 cov=sampled_prior_cov[j])
				cluster_probabilities.append(prob)
			labels[ind] = np.argmax(cluster_probabilities)

		# Get the clusters with our points from the labels
		clusters = [[], [], []]
		for point, label in zip(points, labels):
			clusters[label].append(point)

		probability_guesses.append(deepcopy(sampled_probability_vector))
		mean_guesses.append(deepcopy(sampled_prior_mean))
		covariance_guesses.append(deepcopy(sampled_prior_cov))

		if i % 100 == 0:
			predicted_means = []
			for cl in clusters:
				cl = np.asarray(cl)
				predicted_means.append(np.mean(cl, axis=0))
			
			for i in xrange(len(means)):
				print "Predicted Mean Cluster, ", i, ": ", predicted_means[i]
				print "\n"


	mean_guesses = np.asarray(mean_guesses)
	probability_guesses = np.asarray(probability_guesses)

	print "Posterior Information"
	print "-----------------"
	for i in xrange(len(sampled_prior_mean)):
		print "\ncluster: ", (i + 1)
		print "posterior mean: ", np.mean(mean_guesses[200:], axis=0)[i]
		print "\n"
		print "posterior covariance: ", np.mean(covariance_guesses[200:], axis=0)[i]
		print "\n"
		print "posterior probability: ", np.mean(probability_guesses[200:], axis=0)[0][i]
		print "\n"

	return probability_guesses[200:], mean_guesses[200:], covariance_guesses[200:], clusters, np.mean(mean_guesses[200:], axis=0), sampled_prior_cov, labels


def plot_mean_guesses(actual_means, mean_guesses):
	mean_guesses = np.asarray(mean_guesses)
	f, axarr = plt.subplots(3, sharex=True)
	print len(actual_means), len(mean_guesses)
	for i in xrange(len(actual_means)):
		for j in xrange(len(actual_means[i])):
			x = [k for k in xrange(len(mean_guesses))]
			y = [actual_means[i][j] for k in xrange(len(mean_guesses))]
			axarr[i].plot(x, y, color='black', linewidth=2)

	for i in xrange(mean_guesses.shape[1]):
		x = [k for k in xrange(mean_guesses.shape[0])]
		for j in xrange(mean_guesses.shape[2]):
			axarr[i].plot(x, mean_guesses[:, i, j], color='blue')

	plt.show()


def plot_prob_guesses(prob_guesses, probs):

	plt.title('Cluster Probabilities')
	for i in xrange(len(probs)):
		plt.plot([k for k in xrange(len(prob_guesses))],
				 [float(probs[i])/sum(probs) for k in xrange(len(prob_guesses))],
				 color='black')

	colors = ['green', 'blue', 'red']
	prob_guesses = np.asarray(prob_guesses)

	for i in xrange(prob_guesses.shape[2]):
		plt.plot([k for k in xrange(prob_guesses.shape[0])],
				 prob_guesses[:, 0, i], color=colors[i])

	plt.show()


def l2_error(vector1, vector2):
	return sum([abs(a-b) for a, b in zip(vector1, vector2)])


def confusion_matrix(sample_clusters, predicted_labels):
	conf = np.zeros(shape=(3, 3))

	targets = [0 for i in xrange(sample_clusters[0])]
	targets.extend([1 for i in xrange(sample_clusters[1])])
	targets.extend([2 for i in xrange(sample_clusters[2])])

	print len(targets), len(pred_labels)

	for i in xrange(len(targets)):
		conf[targets[i]][pred_labels[i]] += 1

	print "Confusion matrix: "
	print conf


def generate_all_recreation_plots(prob_guesses, mean_guesses, cov_guesses):
	prob_guesses = np.asarray(prob_guesses)
	mean_guesses = np.asarray(mean_guesses)
	cov_guesses = np.asarray(cov_guesses)

	f, axarr = plt.subplots(3, 3, sharey='row')
	f.suptitle('Mean Values')
	for i in xrange(3):
		for j in xrange(3):
			axarr[i, j].hist(mean_guesses[:, i, j], bins=20)

	plt.show()


	f, axarr = plt.subplots(3, 1)
	f.suptitle('Probabilities')
	for i in xrange(3):
		axarr[i].hist(prob_guesses[:, 0, i], bins=20)

	plt.show()


	f, axarr = plt.subplots(3, 3, sharey='row')
	f.suptitle('Variances')
	for i in xrange(3):
		for j in xrange(3):
			axarr[i, j].hist(cov_guesses[:, i, j, j], bins=20)

	plt.show()


if __name__ == "__main__":

	means = [[4, 0, 2], [0, 1, -1], [0, 0, 0]]
	covariances = [np.identity(3), np.identity(3),
				   np.asarray([[9, 0, 0], [0, 9, 0], [0, 0, 25]])]
	samples_per_cluster = [100, 200, 50]

	data = generate_mvn_data(means, covariances, samples_per_cluster)
	plot_data(data)

	points, km = kmeans(data)

	# Find new centers of three clusters after modifying kmeans output
	new_clusters = [[] for i in xrange(len(means))]
	for point, label in zip(points, km.labels_):
		new_clusters[label].append(point)
	
	predicted_means = []
	for cl in new_clusters:
		cl = np.asarray(cl)
		predicted_means.append(np.mean(cl, axis=0))
	

	# Perform gibbs sampling
	prob_guesses, mean_guesses, cov_guesses, clusters, \
	gibbs_mean, gibbs_cov, pred_labels = \
		gibbs(np.asarray(points), km.labels_)
	plot_mean_guesses(means, mean_guesses)
	plot_prob_guesses(prob_guesses, samples_per_cluster)

	for i in xrange(len(means)):
		print "Actual Mean Cluster, ", i, ": ", means[i]
		print "KMeans Predicted Mean Cluster, ", i, ": ", predicted_means[i]
		print "KMeans Error: ", l2_error(means[i], predicted_means[i])
		print "Gibbs Sampled Mean Cluster, ", i, ": ", gibbs_mean[i]
		print "Gibbs Sampled Error: ", l2_error(means[i], gibbs_mean[i])
		print "\n"


	confusion_matrix(samples_per_cluster, pred_labels)
	generate_all_recreation_plots(prob_guesses, mean_guesses, cov_guesses)


