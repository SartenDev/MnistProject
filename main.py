from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np
import random
from math import exp, log

def relu_derivative(x):
    return 1 if x > 0 else 0
def initialize_weights(input_size, layer_size):
    WagesI = []
    for i in range(layer_size):
        WagesJ = []
        for j in range(input_size):
            WagesJ.append(random.uniform(-1., 1.))
        WagesI.append(WagesJ)
    return WagesI

def relu(sum_values):
    temp = []
    for i in range(len(sum_values)):
        temp.append(max(0, sum_values[i]))
    return temp

def softmax(sum_values):
    temp = []
    expotentials = []
    max_value = max(sum_values)
    for j in sum_values:
        expotentials.append(exp(j - max_value))
    sum_of_expotential = sum(expotentials)
    for l in expotentials:
        temp.append(l / sum_of_expotential)
    return(temp)

def forward_pass(input, output, Layer, Wages):
    sum_values = []
    for i in range(output):
        sum_value = 0
        for j in range(input):
            wage = Wages[i][j]
            sum_value += Layer[j] * wage
        sum_values.append(sum_value)
    return sum_values

def back_prop_error(y, Layer, dense, wages):
    BPE_output_layer = []
    for i in range(0, len(Layer)):
        if i is y:
            BPE_output_layer.append(1 - (Layer[i]))
        else:
            BPE_output_layer.append(Layer[i])
    temp = []
    for i in range(dense):
        sum_of_bpe = 0
        for j in range(len(BPE_output_layer)):
            sum_of_bpe += wages[j][i] * BPE_output_layer[j]
        temp.append(sum_of_bpe)
    return temp, BPE_output_layer

def set_new_wages_relu(BPE_layer, Layer, X, Wages, dense, learning_rate):
    temp = []
    for i in range(dense):
        wages_seperated = []
        for j in range(len(X)):
            wage = Wages[i][j] + learning_rate * BPE_layer[i] * relu_derivative(Layer[i]) * X[j]
            wages_seperated.append(wage)
        temp.append(wages_seperated)
    return temp

def set_new_wages_softmax(Layer, X, Wages, dense2, dense1, learning_rate, epsilon):
    temp = []
    for i in range(dense2):
        wages_seperated = []
        for j in range(dense1):
            wage = Wages[i][j] + learning_rate * -(log(Layer[j] + epsilon))
            wages_seperated.append(wage)
        temp.append(wages_seperated)
    return temp

def predict(X, Wages1, Wages2):
    Layer1 = relu(forward_pass(input=len(X), output=len(Wages1), Layer=X, Wages=Wages1))
    Layer2 = softmax(forward_pass(input=len(Layer1), output=len(Wages2), Layer=Layer1, Wages=Wages2))
    for i in range(len(Layer2)):
        print(i, ".", round(Layer2[i], 3))
    return Layer2.index(max(Layer2))

def run(X_full, y_full, dense1, dense2, learning_rate, epsilon, epochs):
    Wages1 = initialize_weights(len(X_full[0]), dense1)
    Wages2 = initialize_weights(dense1, dense2)
    print(Wages2)
    print(len(Wages1))
    print(len(Wages1[0]))
    for epoch in range(epochs):
        correct_predictions = 0
        for i in range(len(X_full)):
            Layer1 = relu(forward_pass(input=len(X_full[0]), output=dense1, Layer=X_full[i], Wages=Wages1))
            Layer2 = softmax(forward_pass(input=dense1, output=dense2, Layer=Layer1, Wages=Wages2))
            bp_error = back_prop_error(y=y_full[i], Layer=Layer2, dense=dense1, wages=Wages2)
            first_layer_bperror = bp_error[0]

            prediction = Layer2.index(max(Layer2))
            if prediction == y_full[i]:
                correct_predictions += 1

            Wages1 = set_new_wages_relu(BPE_layer=first_layer_bperror, Layer=Layer1,X=X_full[i], Wages=Wages1,dense=dense1, learning_rate=learning_rate)
            TempLayer = relu(forward_pass(input=len(X_full[0]), output=dense1, Layer=X_full[i],Wages=Wages1))
            Wages2 = set_new_wages_softmax(Layer=TempLayer, X=X_full[i], Wages=Wages2, dense2=dense2, dense1=dense1, learning_rate=learning_rate, epsilon=epsilon)

            accuracy = correct_predictions / len(X_full)
            print(f"Epoch: {epoch+1}/{epochs}, Obrazek: {i}, Dokładność: {accuracy:.4f}, Dokładność2: {(correct_predictions/(i+1)):.4f}")


    print("Predicted label:", predict(X=X_full[-2], Wages1=Wages1, Wages2=Wages2))
    print("Actual label:", y_full[-2])


####################################
#                                  #
#               MAIN               #
#                                  #
####################################

mnist = fetch_openml('mnist_784', as_frame=False)

X_full, y_full = mnist.data / 255.0, mnist.target.astype(int)

Wages = []
Wages2 = []
dense1 = 100
dense2 = 10
learning_rate = 0.01
epsilon = 1e-10
epoch = 8

run(X_full, y_full, dense1, dense2, learning_rate, epsilon, epoch)
