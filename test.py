import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from Descision_tree_classifier import DescisionTreeClassifier
from Descision_tree_classifier import Inquiry

# load data
data_url = 'data/train.csv'
names = ['color', 'number', 'species']
dataset = pandas.read_csv(data_url, names=names)
# print(dataset.head(5))
# print(dataset['species'].value_counts())
clas = DescisionTreeClassifier(dataset)

# test partition
t, f = clas.partition_data(dataset, Inquiry('color', 'Red'))
print('PARTITION:')
print('True_branch -->\n', t)
print('False_branch -->\n',f)
print()

# gini test
cur_unce = clas.gini_impurity(dataset)
print('GINI_IMPURITY =', cur_unce)
print()
# clas.print_tree()

# info gain test
t, f = clas.partition_data(dataset, Inquiry('color', 'Green'))
print('INFO_GAIN =', clas.entropy(t, f, cur_unce))
print()

# test best split
r1, r2 = clas.get_best_split(dataset)
print('BEST SPLIT:', r1, r2)
print('CLASSIFICATION:', clas.classify(dataset.loc[1]))
print()

# testing data
data_url = 'data/test.csv'
names = ['color', 'number', 'species']
test_dataset = pandas.read_csv(data_url, names=names)
for i, row in test_dataset.iterrows():
    print('Actual: %s. Predicted %s' % (row[-1], clas.classify(row)))