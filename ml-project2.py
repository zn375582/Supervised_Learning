# Naive Bayes evaluation of heart disease data
# Source referenced: https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/
from pandas import read_csv
from math import sqrt
from math import pi
from math import exp
from random import randrange

# Method to calculate mean of list
def mean(numbers):
    sum = 0
    count = 0
    for num in numbers:
        if num != '?':
            sum = sum + float(num)
            count = count + 1
    return sum/count

# Method to calculate standard deviation of list
def stdev(numbers):
    # Get the mean of the list first
    avg = mean(numbers)
    # Make variables for numerator and denominator for calculations
    numerator = 0
    denominator = 0
    for num in numbers:
        if num != '?':
            numerator = numerator + ((float(num)-avg)**2)
            denominator = denominator + 1
    variance = numerator / (denominator-1)
    return sqrt(variance)


# Calculate the probabilities of predicting each class for a given row
def calculate_diff_outcome_probs(column_info, row):
    # Get number of rows
    num_rows = sum([column_info[label][0][2] for label in column_info])
    probabilities = dict()
    # Classes are whether a person has heart disease or not (0 or 1)
    for class_value, class_info in column_info.items():
        probabilities[class_value] = column_info[class_value][0][2]/float(num_rows)
        for i in range(len(class_info)):
            mean, stdev, len = class_info[i]
            exponent = exp(-((float(row[i])-mean)**2 / (2 * stdev**2 )))
            probabilities[class_value] = probabilities[class_value] * (1 / (sqrt(2 * pi) * stdev)) * exponent
    return probabilities

 
# algorithm to implement naive bayes evaluation
def naive_bayes(train, test):
    #separate dataset by outcome
    dataset_sep = dict()
    # list of patients with no heart disease
    dataset_sep[0] = list()
    # list of patients with heart disease
    dataset_sep[1] = list()
    for i in range(len(train)):
        row = train[i]
        if_heart = int(row[-1])
        # if the patient does not have heart disease
        if if_heart == 0:
            dataset_sep[0].append(row)
        # if the patient does have heart disease i.e. a value of 1, 2, 3, or 4
        else:
           dataset_sep[1].append(row) 
           
    # Create dictionary of info on each column (variable)
    column_info = dict()
    for if_heart, rows in dataset_sep.items():
        summary = [(mean(column), stdev(column), len(column)) for column in zip(*rows)]
        del(summary[-1])
        column_info[if_heart] = summary
        
    predictions = list()
    for row in test:
        probabilities = calculate_diff_outcome_probs(column_info, row)
        best_label, best_prob = None, -1
        for if_heart, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = if_heart
        output = best_label
        predictions.append(output)
    return(predictions)


# This method handles both the cross validation split and calculation of the accuracy for different splits
def evaluate_algorithm(dataset, naive, fold_num):
    #Create copy of the dataset so that data can be popped from the list 
    data_copy = list(dataset)
    # Make fold size equal
    fold_sz = int(len(dataset) / fold_num)
    # Make list to store folds of data
    folds_of_data = list()
    #Do this one time for each fold
    for i in range(fold_num):
        #Create list to store current fold
        fold = list()
        # Stop adding to the fold when the fold size is exceeded
        while len(fold) < fold_sz:
            # Make data in each fold random using randrange to pick a datapoint
            datapoint = randrange(len(data_copy))
            # Add the random datapoint to the existing fold and remove that datapoint so it can't be reused
            fold.append(data_copy.pop(datapoint))
        # Add the finished fold to the list of all folds
        folds_of_data.append(fold)

    results = list()
    for fold in folds_of_data:
        #Copy over all folds into training set
        training_set = list(folds_of_data)
        # Remove current fold from the training set used
        training_set.remove(fold)
        # Add empty list to replace removed fold 
        training_set = sum(training_set, [])
        # Create list for testing set to be inserted into
        testing_set = list()
        # Add elements from current fold to create test set
        for fold_elem in fold:
            testing_set.append(list(fold_elem))
        predicted = naive_bayes(training_set, testing_set)
        actual_vals = list()
        for fold_elem in fold:
            actual_vals.append(fold_elem[-1])
        #Keep count of total correct values
        correct = 0
        # Go through all rows and determine if the predicted value matches the acutal value
        for i in range(len(actual_vals)):
            if actual_vals[i] == predicted[i]:
                # Add to correct count if the prediction matches actual value
                correct += 1
        # accuracy calculation
        acc = correct / float(len(actual_vals)) * 100.0
        results.append(acc)
    return results

# read in dataset from computer
dataset = read_csv('/Users/charlottebailey/Downloads/cleveland_heart_edited.csv')
#convert dataset to list
dataset = dataset.values.tolist()

n_folds = 300
all_acc = evaluate_algorithm(dataset, naive_bayes, n_folds)
print('Accuracy Avg: %.3f%%' % (sum(all_acc)/float(len(all_acc))))
print('Accuracies Across Folds: %s' % all_acc)
