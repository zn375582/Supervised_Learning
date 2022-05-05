
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import sklearn.model_selection as skm

#load data into numpy array
data_x = np.genfromtxt("cleveland_heart_dis_xval_processed.csv", delimiter=",", skip_header=1)
data_y = np.genfromtxt("cleveland_heart_dis_yval.csv", delimiter=",", skip_header=1)

#If y is 1,2,3 then they have heart disease, if 0 then don't
for index, value in enumerate(data_y):
    if value > 1:
        data_y[index] = 1


#remove 20% of the examples and keep them for testing
xtrain, xtest, ytrain, ytest = skm.train_test_split(data_x, data_y, test_size=.2, random_state=34)

#Split the remaining examples into training (75%) and validation (25%)
x_train, x_valid, y_train, y_valid = skm.train_test_split(xtrain, ytrain, test_size=.25, random_state=34)

#Binary Logistic regression fit to entire training set
class LogisticRegression:

    def __init__(self, learning_rate=0.0001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
            # approximate output variable (y) with linear combination of weights and x, plus bias
            linear_model = np.dot(X, self.weights) + self.bias
            # apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))  # derivative w.r.t weights
            db = (1 / n_samples) * np.sum(y_predicted - y)  # derivative w.r.t bias

            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy * 100

itr=[10, 1000, 5000, 10000, 40000, 100000, 1000000, 2000000]
acc=[]

#Fit on training data, predict on validation data
for i in itr:
    lg = LogisticRegression(learning_rate=0.01, n_iters=i)
    lg.fit(x_train, y_train)
    predictions = lg.predict(x_valid)
    print("LR classification accuracy for {} iterations: {}".format(i, accuracy(y_valid, predictions)))
    acc.append(accuracy(y_valid, predictions))

print(lg.weights)
print(lg.bias)

#Plot different number of iterations and their corresponding accuracy
plt.scatter(itr,acc,color="k")
plt.plot(itr,acc)
plt.title("Predict on Validation Set")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.show()


#Fit on training and validation data, predict on test
acc=[]
for i in itr:
    regressor = LogisticRegression(learning_rate=0.01, n_iters=i)
    regressor.fit(xtrain, ytrain)
    predictions = regressor.predict(xtest)
    print("LR classification accuracy for {} iterations: {}".format(i, accuracy(ytest, predictions)))
    acc.append(accuracy(ytest, predictions))

print(regressor.weights)
print(regressor.bias)

#Plot different number of iterations and their corresponding accuracy
plt.scatter(itr,acc,color="k")
plt.plot(itr,acc)
plt.title("Predict on Test Set")
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.show()


