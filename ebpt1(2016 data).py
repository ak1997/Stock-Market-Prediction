from random import randrange
from random import random
from math import exp
import pandas as pd

#Initializing the database
def initialize_dataset(filename):
    da = pd.read_csv(filename)
    crude_oil = da["Crude Oil"].tolist()
    gold = da["Gold"].tolist()
    forex = da["Forex"].tolist()
    sma = da["SMA"].tolist()
    arima = da["ARIMA"].tolist()
    silver=da["Silver"].tolist()
    copper = da["Copper"].tolist()
    ng = da["Natural Gas"].tolist()
    hdata = da["Opening"].tolist()
    dataset = list(zip(crude_oil,gold,silver,forex,copper,ng,sma,arima,hdata))
    for i in range(len(dataset[0]) - 1):
        str_column_to_float(dataset, i)
    dataset = normalize_data(dataset)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row=list(row)
        row[column]=float(row[column])

#Normalizing the dataset using Min-Max Normalization
def normalize_data(dataset):
    minmax= [[min(column), max(column)] for column in zip(*dataset)]
    ndataset=list()
    for row in dataset:
        row=list(row)
        for i in range(len(row)):
            row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
        ndataset.append(row)
    return ndataset

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(int(n_folds)):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if abs(actual[i]-predicted[i])<=0.1:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset,n_folds,l_rate, n_epoch, n_hidden):
    folds = cross_validation_split(dataset, n_folds)
    n_inputs = 8
    n_outputs = 1
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        train_network(network, train_set, l_rate, n_epoch, 1)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = list(row)
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for neuron in layer:
                errors.append(expected - float(neuron['output']))
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        for row in train:
            outputs = forward_propagate(network, row)
            expected = row[-1]
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs


filename="nifty 100 training data.csv"
train_dataset=initialize_dataset(filename)
n_folds = 5
l_rate = 0.5
n_epoch =3000
n_hidden =24
network= evaluate_algorithm(train_dataset,n_folds, l_rate, n_epoch, n_hidden)
filename="nifty 100 testing data.csv"
test_dataset=initialize_dataset(filename)
predicted=scores=list()
for row in test_dataset:
    prediction=predict(network,row)
    predicted.append(prediction)
for i in range(len(predicted)):
    predicted[i]=''.join(map(str,predicted[i]))
    predicted[i]=(float(predicted[i]))
actual=[row[-1] for row in test_dataset]
accuracy = accuracy_metric(actual, predicted)
print("lrate="+str(l_rate)+",nfolds="+str(n_folds)+",n_epoch="+str(n_epoch)+",nhidden="+str(n_hidden))
print(actual)
print(predicted)
print('Mean Accuracy: %s'%accuracy)

