'''
main.py

The main script that runs the classifiers and loads the dataset (IRIS for training)
'''

# load libraries
import pandas
import numpy
# from pandas.plotting import scatter_matrix
# import matplotlib.pyplot as plt
from Decision_tree_classifier import DecisionTreeClassifier
import sys
from k_nearist_neighbors import KNN_CLASSIFY
from k_nearist_neighbors import GET_ACCURACY
from textwrap import TextWrapper
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def header():
    # project introduction.
    for i in range(50):
        print('# ', end='')
    print()
    for j in range(2):
        print('#', end='')
        for i in range(97):
            print(' ', end='')
        print('#')
    print('# ',end='')
    print('CSE 415 Project 3, Option 3 - Supervised Learning: Comparing Trainable Classifiers'.center(96, ' '), end='')
    print('#')
    print('# ',end='')
    print('Akshit Patel, Wanjia Tang'.center(96, ' '), end='')
    print('#')
    for j in range(2):
        print('#', end='')
        for i in range(97):
            print(' ', end='')
        print('#')
    for i in range(50):
        print('# ', end='')
    print()

text_format = TextWrapper(width=97, initial_indent=' ', subsequent_indent=' ')

def information():
    # information on the supported behavior and steps to use.
    welcome = 'Welcome to the interactive session of our option 3. In this session you can ' +\
            'choose to run the given classification methods on a dataset option of choice. ' +\
            'Each classification method will print certain information of its state while training ' +\
            'and finally output its predictions.'
    for text in text_format.wrap(welcome):
        print(text)
    print()
    details = 'This implementation supports 3 classifier choices namely Decision Tree Classifier,' +\
    ' Random Forest Classifier and K-Nearest Neighbor Classification. With each you can change some of its parameters and play around to achieve desired results.'
    for text in text_format.wrap(details):
        print(text)
    print()

# TODO: FINSIH ADDING ALGO OPTIONS AND MEANINGS
def class_info(type='D'):
    des_tree = 'In a descision tree classifier as the name suggest a tree is build from training data and classification of test data is performed by traversing the tree.3 parameters can be specified to tweak the behavior.\n MAX_DEPTH = the maximum depth desired for decision tree \n MIN_RECORDS = the minmum numbers of data rows desired in the leaf of the tree, 1 would imply complete fit of train data.\n seed = number parameter to control the psuedorandomness of data selected for train/test'
    random_forests = 'Random forest work by building forests of decision trees based on different subsets of attributes. parmeters supported are:\n N_ESTIMATORS = number of decision trees to make the forest \n MAX_DEPTH = maximum depth of each tree in forest \n MIN_RECORDS = minmum number of data classes in leaf of each decision tree \n seed = control the psuedo random nature of processing the data \n BAGGING = Using replacement(random data rows) to generate the trees.'
    k_near = 'K nearest neigbor works lazy by finding the neighbors of the data when classifying test set. From a subset of k nearest euclidean neighbors, the mode is the classification result. Supported parameters: \n K = subset of neighbors to use to perform classification, lower the better usually \n seed = control the psuedo random data collection of train/test data'
    if type.upper() == 'K':
        return k_near
    elif type.upper() == 'R':
        return random_forests
    return des_tree

def get_file_url(txt):
    if txt.upper() == 'I':
        return 'data/iris.data'
    elif txt.upper() == 'B':
        return 'data/data_banknote_authentication.csv'
    else:
        return 'data/custom.csv'

def load_data(data_url, data, seed, split=0.75):
    if data.upper() == 'I':
        names = ['sepal length', 'sepal width', 'petal length', 'petal width', 'class']
        dataset = pandas.read_csv(data_url, names=names)
    elif data.upper() == 'B':
        names = ['variance of Wavelet Transformed image', 'skewness of Wavelet Transformed image', 'curtosis of Wavelet Transformed image', 'entropy of image', 'class']
        dataset = pandas.read_csv(data_url, names=names)
    else:
        dataset = pandas.read_csv(data_url)
        labels = ['a' + str(i) for i in range(dataset.shape[1] - 1)]
        labels.append['class']
        dataset.columns = labels
    train_data = dataset.sample(frac=split, random_state=seed) # 75% of current data chosen randomly
    test_data = dataset.drop(train_data.index)
    return train_data, test_data

def load_knn_data(file, train_data, test_data, seed, split=0.75):
    random.seed(a=seed)
    with open(file, 'r') as csvfile:
	    lines = csv.reader(csvfile)
	    data = list(lines)
	    for x in range(len(data)-1):
	        for y in range(4):
	            data[x][y] = float(data[x][y])
	        if random.random() < split:
	            train_data.append(data[x])
	        else:
	            test_data.append(data[x])

def interract():
    while (True):
        print(' DATASET OPTIONS: \n   "I" = iris,\n   "B" = banknote validation,\n   "C" = custom\n   (Q=Quit)')
        data_set = input('Select Data Set: ')
        
        if data_set == 'Q':
            exit()
        if data_set.upper() not in ('I', 'B', 'C'):
            print('Pls Enter a valid option!')
            continue
        
        split_size = input('What percent of dataset is train? (Default: 75% training, 25% test): ')
        if split_size == '':
            split_size = 0.75
        else:
            split_size.replace('%', '')
            split_size = float(split_size) / 100.0
        data_url = get_file_url(data_set)
        print()
        while True:
            print('CLASSIFIER OPTIONS: \n   "D" = decision tree classifier(default),\n   "K" = K-nearest neighbor,\n   "R" = Random Forest\n   (Q=Quit)\n   (cd=change dataset)')
            class_choice = input('Choose a classifier: ')
            if class_choice.upper() == 'Q':
                exit()
            elif class_choice.upper() == 'CD':
                break
            elif class_choice.upper() not in ('D', 'R', 'K'):
                class_choice = 'D'
            print()
            print(class_info(class_choice)) # print classifier details
            print()
            if class_choice == 'D': # Decision Tree
                max_depth = input('desired MAX_DEPTH (default complete data fit): ')
                max_depth = None if max_depth == '' else int(max_depth)
                min_record = input('desired MIN_RECORDS parameter (default 1): ')
                min_record = None if min_record == '' else int(min_record)
                seed = input('desired seed for random train/test data selection(default=None): ')
                seed = None if seed.upper() in ('', 'NONE') else int(seed)
                train_data, test_data = load_data(data_url, data_set, seed, split_size) # loads the data
                print('Recursively building the Tree....')
                des_tree = DecisionTreeClassifier(train_data, max_depth, min_record) # build the tree
                # print(des.print_tree())
                # test results
                print('Gathering Classification results....')
                predictions = des_tree.classify(test_data)
                print()
                print('Test Accuracy: {:10.2f}%'.format(des_tree.prev_test_accuracy))
                summary = pandas.crosstab(test_data.iloc[:, -1], predictions, margins=True)
                summary.columns.name = 'Predicted →'
                summary.index.name = 'Actual ↴'
                print('Summary:')
                print(summary)
            elif class_choice == 'K': # K-nearest-neighbor
                k = input('Desired K value (default 3): ')
                k = 3 if k == '' else int(k)
                seed = input('desired seed for random train/test data selection(default=None): ')
                seed = None if seed.upper() in ('', 'NONE') else int(seed)
                train_data = []
                test_data = []
                load_knn_data(data_url, train_data, test_data, seed, split_size)
                print('Creating neighbors and will start classification soon after....')
                predictions = KNN_CLASSIFY(train_data, test_data, k)
                test_data_class = numpy.array([value[-1] for value in test_data])
                print()
                print('Test Accuracy: {:10.2f}%'.format(GET_ACCURACY(test_data, predictions)))
                summary = pandas.crosstab(test_data_class, numpy.array(predictions), margins=True, margins_name='Total')
                summary.columns.name = 'Predicted →'
                summary.index.name = 'Actual ↴'
                print('Summary:')
                print(summary)
            else:
                n_estimators = input('desired N_ESTIMATORS? (default 10)')
                n_estimators = 10 if n_estimators.upper() in ('', 'DEFAULT') else int(n_estimators)
                max_depth = input('desired MAX_DEPTH (default complete data fit): ')
                max_depth = None if max_depth.upper() in ('', 'DEFAULT') else int(max_depth)
                min_record = input('desired MIN_RECORDS parameter (default 1): ')
                min_record = 1 if min_record.upper() in ('', 'DEFAULT') else int(min_record)
                bagging = input('Do bagging of train set?(Y/N) (Default Y) ')
                bagging = True if bagging.upper() in ('', 'DEFAULT', 'Y') else False 
                seed = input('desired seed for random train/test data selection(default=None): ')
                seed = None if seed.upper() in ('', 'NONE', 'DEFAULT') else int(seed)
                train_data, test_data = load_data(data_url, data_set, seed, split_size) # loads the data
                
                # classifier
                trained_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=min_record, bootstrap=bagging, random_state=seed)
                print('Building the Forest....')
                trained_model.fit(train_data[train_data.columns[0:train_data.shape[1] - 1]], train_data[train_data.columns[-1]])

                # test data
                print('Gathering Classification results....')
                predicitons = trained_model.predict(test_data[test_data.columns[0:test_data.shape[1]-1]])
                print()
                # print('Train Accuracy: ', accuracy_score(train_data.iloc[:, -1], trained_model.predict(train_data[train_data.columns[0:train_data.shape[1]-1]])))
                print('Test Accuracy: {:10.2f}%'.format(accuracy_score(test_data.iloc[:, -1], predicitons)*100))
                summary =pandas.crosstab(test_data.iloc[:, -1], predicitons)
                summary.columns.name = 'Predicted →'
                summary.index.name = 'Actual ↴'
                print('Summary:')
                print(summary)
            print()
'''
train_test_split() to find the test and train data
'''

def print_project():
    header()
    print()
    information()
    interract()

print_project()
'''


# # shape
# # print(dataset.shape)

# # # head
# # print(dataset.head(20))

# # # descriptions
# # print(dataset.describe())

# # class distribution
# # print(dataset.groupby('class').size())

# # histograms
# # dataset.hist()
# # plt.show()

# TODO: gather statistics, Benchmark details, confusion matrix, etc.
'''
