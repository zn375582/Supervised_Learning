# Decision trees implementation of heart disease data
# Source Referenced: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
from random import seed
from random import randrange
from pandas import read_csv

# This method handles both the cross validation split and calculation of the accuracy for different splits
def evaluate_algorithm(dataset, dec_tree, fold_num, maximum_tree_depth, minimum_sample_split):
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
        predicted = dec_tree(training_set, testing_set, maximum_tree_depth, minimum_sample_split)
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

# Method to calculate the gini index
def gini_index(children, classes):
    gini = 0.0
    # count total number of points to be separated at the node
    num_to_sort = float(sum([len(child) for child in children]))
    for child in children:
        total = 0.0
        size = float(len(child))
        # don't divide by zero
        if size == 0:
            continue
        for ind_class in classes:
            #calculate total by calculating p values
            p = [elem[-1] for elem in child].count(ind_class) / size
            total = total + p * p
        # weight by size
        gini += (1.0 - total) * (size / num_to_sort)
    return gini

# Method to make terminal node
def terminal(node):
    total = list()
    for elem in node:
        total.append(elem[-1])
    return max(set(total), key=total.count)

# Go through possible split points for the dataset and pick the best one
def pick_best_split(dataset):
    #initialize values for comparison
    node_children, node_index, node_value, node_score = None, 999, 999, 999
    classes = list(set(datapoint[-1] for datapoint in dataset))
    for i in range(len(dataset[0])-1):
        for datapoint in dataset:
            left_node = list()
            right_node = list()
            # Split based on the datapoint; if the element is less than the datapoint, it goes to the left node; if more, it goes to the right node
            for elem in dataset:
                if elem[i] < datapoint[i]:
                    left_node.append(elem)
                else:
                    right_node.append(elem)
            children = left_node, right_node
            gini = gini_index(children, classes)
            # If a better gini impurity score is obtained, update all values for the node
            if gini < node_score:
                node_index = i
                node_value = datapoint[i]
                node_score = gini
                node_children = children
    return {'index':node_index, 'value':node_value, 'children':node_children}

# Determine fate of nodes: split into children or make it terminal, essentially create the tree based on params
def split(node, maximum_tree_depth, minimum_sample_split, depth):
    left, right = node['children']
    # Remove children from the given node
    del(node['children'])
    # check if needs to be terminal
    if not left or not right:
        node['left'] = node['right'] = terminal(left + right)
        return
    # make terminal if maximum tree depth is reached already
    if depth >= maximum_tree_depth:
        node['left'], node['right'] = terminal(left), terminal(right)
        return
    # split the left child as long as its length exceeds the minimum required 
    if len(left) <= minimum_sample_split:
        node['left'] = terminal(left)
    else:
        node['left'] = pick_best_split(left)
        split(node['left'], maximum_tree_depth, minimum_sample_split, depth+1)
    # split the right child as long as its length exceeds the minimum required
    if len(right) <= minimum_sample_split:
        node['right'] = terminal(right)
    else:
        node['right'] = pick_best_split(right)
        split(node['right'], maximum_tree_depth, minimum_sample_split, depth+1)

# make a prediction based on the tree
def predict(tree, datapoint):
    if datapoint[tree['index']] < tree['value']:
        if isinstance(tree['left'], dict):
            return predict(tree['left'], datapoint)
        else:
            return tree['left']
    else:
        if isinstance(tree['right'], dict):
            return predict(tree['right'], datapoint)
        else:
            return tree['right']

# Decision Tree Algorithm
def decision_tree(training_set, testing_set, maximum_tree_depth, minimum_sample_split):
    tree = pick_best_split(training_set)
    split(tree, maximum_tree_depth, minimum_sample_split, 1)
    predictions = list()
    for row in testing_set:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)

seed(1)
# get dataset
dataset = read_csv('/Users/charlottebailey/Downloads/cleveland_heart_disease_edited_p2.csv')
dataset = dataset.values.tolist()

# decide on parameter values and evaluate algorithm
fold_num = 5
maximum_tree_depth = 20
minimum_sample_split = 10

all_acc = evaluate_algorithm(dataset, decision_tree, fold_num, maximum_tree_depth, minimum_sample_split)
print('Accuracy Avg: %.3f%%' % (sum(all_acc)/float(len(all_acc))))
print('Accuracy across Diff Fold Uses: %s' % all_acc)
