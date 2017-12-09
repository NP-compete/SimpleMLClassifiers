#!/usr/bin/python3.5

import csv, math, random
from math import exp, sqrt, pi

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

def mean(vector):
	""" This function returns the mean of a input vector. """
	return sum(vector) / float(len(vector))

def std_dev(vector):
	""" This function returns the standard deviation of an input vector. """
	if len(vector) == 1:
		return 0.0
	avg = mean(vector)
	vec = [((x - avg) ** 2) for x in vector]
	return sqrt(sum(vec) / float(len(vec) - 1))

def seperate_labeled(vector, labels):
	""" This function return a dict like object. To know about information
	format please refer the documentation of seperated_by_class
	function. Type1 format is the format return by this function. """
	set_labels = list(set(labels))
	set_vector = list(set(vector))
	temp_result = [0] * len(set_labels)
	temp_result = dict(zip(set_labels, temp_result))
	result = {}
	for i in range(len(set_vector)):
		result[set_vector[i]] = temp_result
	for i in range(len(vector)):
		lbl, tpl = labels[i], vector[i]
		result[tpl][lbl] = result[tpl][lbl] + 1
	return result

def seperate_numeral(vector, labels):
	""" This function return a dict like object. To know about information
	format please refer the documentation of seperated_by_class
	function. Type2 format is the format return by this function. """
	set_labels = list(set(labels))
	result = {}
	for i in range(len(vector)):
		temp = result.get(labels[i], [])
		temp.append(vector[i])
		result[labels[i]] = temp
	for x in result.keys():
		result[x] = (mean(result[x]), std_dev(result[x]))
	return result

def seperated_by_class(dataset, labeled_columns=[]):
	""" This function returns a list object of dict objects corresponding to
	every column in dataset which contains the information of two types :
	
	1) If the column contains labeled data then dictionary contains
	information in given format
	{col_value1 : {class_label1: count1, class_label2: count2},
	 ....
	 ....
	 ....
	 col_valuen : {class_label1: counti, class_label2: countj}
	}

	2) If the column contains numerical data then dictionary contains
	information in given format
	{class_label1: (mean1, std_dev1), .....class_labeln: (meann, std_devn)} """
	result = []
	for i in range(len(dataset[0]) - 1):
		if dataset[0][i] in labeled_columns:
			result.append(seperate_labeled(dataset[1][i], dataset[1][-1]))
		else:
			result.append(seperate_numeral(dataset[1][i], dataset[1][-1]))
	return result

def class_prob(labels):
	""" This function returns a dict object which has keys as class
	names and corresponding value is the class probablity. """
	result = dict(zip(set(labels), [0] * len(set(labels))))
	for label in labels:
		result[label] = result[label] + 1
	for label, val in result.items():
		result[label] = val / float(len(labels))
	return result

def get_prob(x, tpl):
	""" This function returns the probablity of a number
	by using uniform probablity distribution function. """
	mean, std = tpl
	ret = exp(-0.5 * (((x - mean) / std) ** 2))
	ret = (ret * sqrt(0.5 / pi)) / std
	return ret

def predict(test_data, seperated, classes):
	""" This function predict the class of test data tupples by
	calculating and comparing the probablity of belonging to
	a specific class and return a tuple. First element of
	which is a list of predicted labels and another element
	is the list of probablity distribution of classes in data. """
	result, pred_cls = [], []
	for test in test_data:
		temp, maxp, maxc = [], -1, None
		for cls, prob in classes.items():
			for i in range(len(test)):
				if isinstance(test[i], str):
					prob = prob * seperated[i][test[i]][cls]
				elif seperated[i][cls][1] > 0:
					prob = prob * get_prob(test[i], seperated[i][cls])
				elif seperated[i][cls][0] == 0:
					prob = 0
			if prob > maxp:
				maxp = prob
				maxc = cls
			temp.append(prob)
		result.append(temp)
		pred_cls.append(maxc)
	return pred_cls, result

def accuracy_score(predictions, actual):
	""" This function returns the accuracy score of the model
	by cross validating the predictions given by model.  """
	count, total = 0.0, len(actual)
	for i in range(total):
		if predictions[i] == actual[i]:
			count += 1
	return count / total

def main():
	""" Driver function to test Gaussian Naive Bayes Classifier. """ 
	# Read data from file with column headings
	dataset = read_csv("iris.data")
	# dataset = read_csv("abalone.data")

	# Apply train test split to train and validate model 
	dataset, test_data = train_test_split(dataset, split_ratio = 0.2)
	# dataset, test_data = train_test_split(dataset, split_ratio = 0.1, random_state = False)

	# Change the dataset orientation from row to column wise
	dataset = (dataset[0], list(zip(*dataset[1])))

	# Seperate Columns according to class labels 
	seperated = seperated_by_class(dataset)
	# seperated = seperated_by_class(dataset, ["Sex"])

	# Get class probablity
	classes = class_prob(dataset[1][-1])

	# Make predictions
	predictions = predict(test_data[0], seperated, classes)

	# Get accuracy score of Model
	print(accuracy_score(predictions[0], test_data[1]))

	# for i in range(len(predictions[0])):
	# 	print(predictions[0][i], test_data[1][i])

if __name__ == "__main__":
	main()