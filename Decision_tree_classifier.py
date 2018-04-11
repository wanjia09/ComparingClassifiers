'''
CART type decision tree

1) Add root node of the tree
    - ALL Nodes recieve a list of rows for input, 
    - ROOT gets entire training set
    - Each node asks a true or false question on features/attributes
        in response to the data we split/partition the data into two 
        subsets, GOAL TO UNMIX THE LABELS as we proceed down the tree
    - Which question to ask when to build a better tree. To do this we must quantify
        the uncertainy of asking the question using GINI IMPURITY at single node.
        Quantify how much the question reduces the uncertainty using INFORMATION GAIN
    - left node is unmixed i.e quantity know
    - righ node is mixed i.e. we must further resolve
    2) what questions to ask?
        - iterate over every value for every attribute to get possible insight.
        - 
'''
import numpy
from pandas.api.types import is_number
from pandas import Series

class DecisionTreeClassifier:

    def __init__(self, dataset, max_depth, min_records):
        self.dataset = dataset
        self.max_depth = max_depth
        self.min_records = min_records
        self.prev_test_accuracy = 0
        self.tree = self.__build_tree(dataset, max_depth, min_records)

    def class_counts(self, data):
        # returns the count of classifier label, assumes label at last index in dataset
        return data[data.columns[-1]].value_counts()

    def partition_data(self, data, inquiry):
        '''
        partitions this dataset into yes and no evaluated datasets of the given inquiry
        return yes_data, no_data
        '''
        lab = inquiry.label
        val = inquiry.value
        if is_number(val):
            yes_data = data[data[lab] >= val]
        else:
            yes_data = data[data[lab] == val]
        no_data = data.drop(yes_data.index)
        return yes_data, no_data

    def gini_impurity(self, data):
        '''Calculate the Gini Impurity for a list of rows.

        See: https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
        '''
        counts = self.class_counts(data)
        impurity = 1
        for col_name, val in counts.iteritems():
            prob_label = val / float(data.shape[0])
            impurity -= prob_label ** 2
        return impurity

    def entropy(self, left_child, right_child, curr_uncertainty):
        '''
        the information gain of the branch given.
        uncertain starting - weighted impurity of both both children
        '''
        len_left = left_child.shape[0]
        len_right = right_child.shape[0]
        p = float(len_left / (len_left + len_right))
        return curr_uncertainty - (p * self.gini_impurity(left_child)) - ((1 - p) * self.gini_impurity(right_child))

    def get_best_split(self, data):
        '''
        Find the best question to ask by iterating over every feature / value
        and calculating the information gain. (change)
        '''

        best_gain = 0  # keep track of the best information gain
        best_question = None  # keep train of the feature / value that produced it
        current_uncertainty = self.gini_impurity(data)

        for col_name, col_series in data.iloc[:, :-1].iteritems(): # iterate for every feature except the last
            values = col_series.unique() # get unique values
            for val in values: # for every unique value calculate the information gain
                # question to be asked for current feautre
                question = Inquiry(col_name, val)
                true_data, false_data = self.partition_data(data, question)

                # Skip the split if it doesn't divide the dataset i.e. we have a probable leaf.
                if true_data.shape[0] == 0 or false_data.shape[0] == 0:
                    continue

                # Calculate the information gain from this split
                gain = self.entropy(true_data, false_data, current_uncertainty)

                # TODO: decide if > or >=
                if gain > best_gain:
                    best_gain, best_question = gain, question
        return best_gain, best_question

    def __build_tree(self, data, max_depth, min_records):
        '''
        recursively builds the descion tree of the training dataset given
        '''
        # Try partitioing the dataset on each of the unique attribute,
        # calculate the information gain,
        # and return the question that produces the highest gain.
        gain, question = self.get_best_split(data)

        # Base case: no further info gain, reached max depth or achieved required record
        # Since we can ask no further questions,
        # we'll return a leaf.
        if gain == 0 or (max_depth is not None and max_depth == 0) or (min_records is not None and data.shape[0] <= min_records):
            return Node(data)

        # If we reach here, we have found a useful feature / value
        # to partition on.
        true_rows, false_rows = self.partition_data(data, question)

        if max_depth: # not none scenario
            max_depth -= 1
        # Recursively build the true branch.
        true_branch = self.__build_tree(true_rows, max_depth, min_records)

        # Recursively build the false branch.
        false_branch = self.__build_tree(false_rows, max_depth, min_records)

        # Return a Question node.
        # This records the best feature / value to ask at this point,
        # as well as the branches to follow
        # depending on the answer.
        return Node(question, true_branch, false_branch)

    def build_tree(self, max_depth=None, min_records=None):
        # recursively build the decision tree. if max_dept=None of min_record=None, then complete data fit.
        return self.__build_tree(self.dataset, max_depth, min_records)

    def print_tree(self):
        self.__print_tree(self.tree)
    
    def __print_tree(self, node, spacing=""):
        if node.is_leaf():
            print(spacing + 'Predict', node.data)
        else:
            print(spacing + str(node.data))

            print(spacing + '--> True:')
            self.__print_tree(node.left, spacing + '    ')

            print(spacing + '--> False:')
            self.__print_tree(node.right, spacing + '    ')

    def classify(self, test_data):
        predictions = []
        correct_items = 0
        for i, row in test_data.iterrows():
            predict_item = self.classify_row(row)
            if predict_item == row['class']:
                correct_items += 1
            predictions.append(predict_item)
        self.prev_test_accuracy = float(correct_items) / test_data.shape[0] * 100
        return numpy.array(predictions)

    def classify_row(self, data_row):
        classification_result = self.__classify(data_row, self.tree)
        # return result in a nicer format
        total = float(classification_result.sum())
        confidence = {}
        for label, val in classification_result.iteritems():
            confidence[label] = int(val) / total * 100
        max_attribute = (None, 0) # attribute that has the best confidence for the data
        for key in confidence:
            confidence_score = confidence[key]
            if confidence_score > max_attribute[1]:
                max_attribute = (key, confidence_score)
        return max_attribute[0]

    def __classify(self, data_row, node):
        '''
        returns the classification of the data_row using the trained data
        '''
        # base case: we have a classification
        if node.is_leaf():
            return node.data
        # return the appropriate true-branch or the false-branch by 
        # comparing the value stored in the node of a particular feature,
        # to the data_row we're considering.
        if node.data.get_ans(data_row):
            return self.__classify(data_row, node.left)
        else:
            return self.__classify(data_row, node.right)


class Node:

    def __init__(self, data, left=None, right=None):
        '''
        data is the dataframe of a potential leaf node or its is a inquiry object for decision node
        two types of node:
            type 1: decision node i.e the inquiry node with left and right branches
            type 2: leaf node i.e. final classification so no children by default
        '''
        self.data = data if isinstance(data, Inquiry) else data[data.columns[-1]].value_counts()
        self.left = left
        self.right = right

    def is_leaf(self):
        return (not isinstance(self.data, Inquiry)) and self.left == None and self.right == None


class Inquiry:
    ''' 
    An inquiry is used to partition a dataset based on the value and label.

    '''
    def __init__(self, label, value):
        self.label = label
        self.value = value

    def get(self):
        return lambda x: self.get_ans(x)  

    def get_ans(self, query_data):
        '''
        compares the value of this Inquiry to the value of query_data based 
        on the label

        query_data is the row of values for which answer to this inquiry is required

        comparison criteria:
            for number intances >= comparison,
            for literal == comparision
            example:
                query_data = [255, 0, 0, 'red']
                obj_1 = {label='R', value='255'}
                obj_1.get_ans(query_data) <- query_data['r'] >= value

                obj_2 = {label='color', value='red'}
                obj_2.get_ans(query_data) <- query_data['color'] == value
        '''
        if is_number(self.value):
            return query_data[self.label] >= self.value
        return query_data[self.label] == self.value

    def __rep__(self):
        condition = '>=' if is_number(self.value) else '=='
        return 'Is %s %s %s?' % (str(self.label), condition, str(self.value))
    
    def __str__(self):
        condition = '>=' if is_number(self.value) else '=='
        return 'Is %s %s %s?' % (str(self.label), condition, str(self.value))
