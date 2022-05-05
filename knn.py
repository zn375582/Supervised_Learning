# source: https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/
import math
from cmath import sqrt
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import sklearn.model_selection as skm
from random import seed
from random import randrange
from csv import reader
import matplotlib.pyplot as plt # plotting

data = np.genfromtxt("full_heartDisease_dataset.csv", delimiter=",", skip_header=1)


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader)  # skip title column
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


data_x = load_csv("cleveland_heart_dis_xval_processed.csv")
data_y = load_csv("cleveland_heart_dis_yval.csv")

# If y is 1,2,3 then they have heart disease, if 0 then don't
for i in range(len(data_y)):
    if int(data_y[i][0]) >= 1:
        data_y[i][0] = 1
    else:
        data_y[i][0] = 0

# remove 20% of the examples and keep them for testing
xtrain, xtest, ytrain, ytest = skm.train_test_split(data_x, data_y, test_size=.2, random_state=34)

# Split the remaining examples into training (75%) and validation (25%)
x_train, x_valid, y_train, y_valid = skm.train_test_split(xtrain, ytrain, test_size=.25, random_state=34)

# append x and y values together
# Format data in set to floats
for i in range(len(xtrain)):
    xtrain[i].append(ytrain[i][0])
    for index in range(len(xtrain[i]) - 1):
        xtrain[i][index] = float(xtrain[i][index])

for i in range(len(xtest)):
    xtest[i].append(ytest[i][0])
    for index in range(len(xtest[i]) - 1):
        xtest[i][index] = float(xtest[i][index])

# For second split
for i in range(len(x_train)):
    x_train[i].append(y_train[i][0])
    for index in range(len(x_train[i]) - 1):
        x_train[i][index] = float(x_train[i][index])

for i in range(len(x_valid)):
    x_valid[i].append(y_valid[i][0])
    for index in range(len(x_valid[i]) - 1):
        x_valid[i][index] = float(x_valid[i][index])

# Reformat y values for comparison with predictions later
for i in range(len(ytrain)):
    ytrain[i] = ytrain[i][0]

for i in range(len(ytest)):
    ytest[i] = ytest[i][0]

for i in range(len(y_train)):
    y_train[i] = y_train[i][0]

for i in range(len(y_valid)):
    y_valid[i] = y_valid[i][0]


# Calculate the Euclidean Distance
def euclidean_dist(row1, row2):
    dist = 0.0
    for i in range(len(row1) - 1):
        dist += math.pow((row2[i] - row1[i]), 2)
    dist = math.pow(dist, .5)
    return dist


# Get the nearest neighbors
def get_neighbors(train, test_row, num_nbr):
    distances = list()
    for train_row in train:
        dist = euclidean_dist(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    nbr = list()
    for i in range(num_nbr):
        nbr.append(distances[i][0])
    return nbr


# Predict
def predict(train, test_row, num_nbr):
    nbr = get_neighbors(train, test_row, num_nbr)
    output_val = [row[-1] for row in nbr]
    prediction = max(set(output_val), key=output_val.count)
    return prediction


# kNN Algorithm
def k_nearest_neighbors(train, test, num_nbr):
    predictions = list()
    for row in test:
        output = predict(train, row, num_nbr)
        predictions.append(output)
    return (predictions)


# Find accuracy of algorithm
# Calculate accuracy percentage
def accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


# Test the kNN on the heart data set
# evaluate algorithm with train and validation data, against test set
num_neighbors = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 27]
acc = []
# experiment with k values

k = 17
predictions = k_nearest_neighbors(xtrain, xtest, k)
# Check accuracy
accuracy_percentage = accuracy(ytest, predictions)
acc.append(accuracy_percentage)
print("Accuracy for {0}-nearest neighbors: {1}".format(k, accuracy_percentage))

k = 19
predictions = k_nearest_neighbors(xtrain, xtest, k)
# Check accuracy
accuracy_percentage = accuracy(ytest, predictions)
acc.append(accuracy_percentage)
print("Accuracy for {0}-nearest neighbors: {1}".format(k, accuracy_percentage))

'''
# Plot different number of k values and their corresponding accuracy
plt.scatter(num_neighbors,acc,color="k")
plt.plot(num_neighbors,acc)
plt.title("Predict on Test Set")
plt.xlabel("k-value")
plt.ylabel("Accuracy")
plt.show()

print("\n")
'''
acc = []
# evaluate alg by training with train set and predicting on validation set
for k in num_neighbors:
    predictions = k_nearest_neighbors(x_train, x_valid, k)
    # Check accuracy
    accuracy_percentage = accuracy(y_valid, predictions)
    acc.append(accuracy_percentage)
    print("Accuracy for {0}-nearest neighbors: {1}".format(k, accuracy_percentage))

# Plot different number of k values and their corresponding accuracy
plt.scatter(num_neighbors,acc,color="k")
plt.plot(num_neighbors,acc)
plt.title("KNN: Train-S1, Predict-S2")
plt.xlabel("k-value")
plt.ylabel("Accuracy")
plt.show()