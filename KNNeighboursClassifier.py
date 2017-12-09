#!/usr/bin/python3.5


import math, csv, random
from math import pow, sqrt
import operator as op

def read_csv(file_name):
	""" Utility function to read csv files, and csv files must
	have headers defined in them otherwise it may raise an
	exception. It returns a tupple, first element of which
	is the column headers and second element is a list of 
	data tupples specified in csv file (row wise). """
	dataset = list(csv.reader(open(file_name, "r")))
	for i in range(len(dataset)):
		result = []
		for j in range(len(dataset[i])):
			try:
				dataset[i][j] = float(dataset[i][j])
			except:
				pass
	return dataset[0], dataset[1:]

def encode_column(dataset, column, mapping):
	""" This is a utility function which encode the values of a
	specific column using dictionary named mapping."""
	idx = dataset[0].index(column)
	for i in range(len(dataset[1])):
		dataset[1][i][idx] = mapping[dataset[1][i][idx]]

def rename_column(dataset, old_column, new_column):
	""" This is a utility function which renames the name of a specific column. """
	idx = dataset[0].index(old_column)
	dataset[0][idx] = new_column

def drop_column(dataset, column):
	""" This function drops the specific column from the dataset. """
	idx = dataset[0].index(column)
	dataset[0].pop(idx)
	for i in range(len(dataset[1])):
		dataset[1][i].pop(idx)

def train_test_split(dataset, split_ratio = 0.25, random_state = True):
	""" This function is an utility function to split data for
	trainning and testing purpose. It returns a tuple which
	in turn are the tuples. First element of return value is
	trainning dataset its first element contains name of
	headers while second element contains trainning data.
	Second element is test dataset, its first element contains
	validation dataset (data without labels) and second element
	contains all labels corresponding to test dataset. """
	test, length, i = ([], []), len(dataset[1]), 0
	while len(dataset[1]) > split_ratio * length:
		if random_state:
			i = random.randrange(0, len(dataset[1]))
		e = dataset[1].pop(i)
		test[0].append(e[:-1])
		test[1].append(e[-1])
	return dataset, test

def euclidean_distance(inst1, inst2):
	""" This function returns the euclidean distance between 2 vectors"""
	d = 0
	for i in range(len(inst1)):
		d += pow((inst1[i] - inst2[i]), 2)
	return sqrt(d)

def get_neighbours(point, dataset, K):
	""" This function returns a list of labels of first K vectors from
		trainning set which are nearest to the input param point. """
	result, neighbours = [], []
	for i in range(len(dataset)):
		result.append((dataset[i], euclidean_distance(point, dataset[i][:-1])))
	result = sorted(result, key=op.itemgetter(1), reverse=False)
	for i in range(K):
		neighbours.append(result[i][0][-1])
	return neighbours

def predict(neighbours):
	""" This function returns a tuple object first element of tuple is the label
		of predicted class and another element is a list of votes of labels. """
	set_neighbours = list(set(neighbours))
	votes = [set_neighbours.count(n) for n in set_neighbours]
	res, cls = -1, None
	for i in range(len(set_neighbours)):
		if votes[i] > res:
			res = votes[i]
			cls = set_neighbours[i]
	return cls, list(zip(set_neighbours, votes))

def accuracy_score(predictions, actual):
	""" This function returns the accuracy score of model."""
	correct = 0.
	for i in range(len(actual)):
		if predictions[i] == actual[i]:
			correct += 1.
	return correct / float(len(actual))

def main():
	""" Driver function to test KNN Classifier. """ 
	# Read data from file with column headings
	dataset, predictions = read_csv("iris.data"), []
	# dataset, predictions = read_csv("abalone.data"), []
	# encode_column(dataset, "Sex", {'M': 0, 'F': 1, 'I': 2})
	
	# Apply train test split to train and validate model 
	dataset, test_data = train_test_split(dataset, split_ratio = 0.2)

	# Make predictions for every test instance.
	for i in range(len(test_data[0])):
		neighbours = get_neighbours(test_data[0][i], dataset[1], 3)
		prediction = predict(neighbours)
		predictions.append(prediction[0])

	# Get the accuracy of the model
	print(accuracy_score(predictions, test_data[1]))

if __name__ == "__main__":
	main()