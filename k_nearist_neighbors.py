'''
This file implements the K nearest neighbor classifier.
'''
import math
import operator

def calculate_similarity(item1, item2, length_parameter):
    '''calculate and return the euclidean distance of item1 and item2, which are presented as lists.'''
    distance = 0
    for i in range(length_parameter):
        distance += pow((item1[i] - item2[i]), 2)
    return math.sqrt(distance)

def get_neighbors(trainingset, given_item, k):
    '''for a training set and a given item, calculate all the similarities of the items in the set and sort them from most similiar to least.'''
    distance = []
    length = len(given_item) - 1
    for i in range(len(trainingset)):
        distance_of = calculate_similarity(trainingset[i], given_item, length)
        distance.append((trainingset[i], distance_of))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distance[i][0])
    return neighbors

def make_predictions(neighbors):
    '''evaluate all the neighbors and return the most similiar ones.'''
    nearest_neighbors = {}
    for i in range(len(neighbors)):
        response = neighbors[i][-1]
        nearest_neighbors[response] = nearest_neighbors.get(response, 0) + 1
    sorted_neighbors = sorted(nearest_neighbors.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_neighbors[0][0]

def getAccuracy(testData, predictions):
    '''test the predictions on the test data.'''
    correct_items = 0
    for i in range(len(testData)):
        if (testData[i][-1]) == predictions[i]:
            correct_items += 1
    return (correct_items/float(len(testData))) * 100.0

def classify(train_data, test_data, k=3):
    # get predictions
    predictions = []
    for row in range(len(test_data)):
        neighbors = get_neighbors(train_data, test_data[row], k)
        result = make_predictions(neighbors)
        predictions.append(result)
        # print('actual=' + repr(result) + ' predicted=' + repr(test_data[row][-1]))
    # print('Accuracy', getAccuracy(test_data, predictions), end='%')
    return predictions

# lambda function to support initializing the random_tree_classifier
KNN_CLASSIFY = lambda train_data, test_data, k: classify(train_data, test_data, k)
GET_ACCURACY = lambda train_data, predictions: getAccuracy(train_data, predictions)
