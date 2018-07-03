import math
import random

class LinearRegressor:
    """
    A simple class that implements linear regression
    
    Methods:
    predict()
    fit() 
    """

    def __init__(self):
        """Initialize the parameters randomly"""
        # randomly initialize parameters b/w 0 to 1
        self.x0 = random.random()
        self.x1 = random.random()

    def predict(self, input1):
        """Return the output for a given input"""
        # return the output
        return self.x0 + self.x1 * input1

    def fit(self, inputs, outputs, learning_rate=0.00001, iterations=10000):
        """
        Train the model

        Keyword Arguments:
        inputs -- 
        outputs -- 
        learning_rate --
        iterations --
        """
        no_of_training_examples = len(inputs)

        # gradient descent algorithm
        while iterations > 0:
            
            # Use Ordinary least squares regression model
            predictions = [self.predict(i) for i in inputs]
            # print(predictions)
            derivative_x0 = sum([predictions[i] - outputs[i]
                                 for i in range(no_of_training_examples)]) 
            # print(derivative_x0)
            derivative_x1 = sum([(predictions[i] - outputs[i]) * inputs[i]
                                 for i in range(no_of_training_examples)]) 
            # print(derivative_x1)
            self.x0 = self.x0 - learning_rate * derivative_x0
            self.x1 = self.x1 - learning_rate * derivative_x1
            iterations -= 1

LR = LinearRegressor()
inputs = [1, 2, 5, 10, 15, 30, 45]
outputs = [1, 4, 10, 20, 30, 60, 90]
# inputs = [(i - min(inputs)) / (max(inputs) - min(inputs)) for i in inputs]
# outputs = [(i - min(outputs)) / (max(outputs) - min(outputs)) for i in outputs]
LR.train(inputs, outputs, iterations=10000)
print('For input: %.2f, prediction: %.2f' % (0.5, LR.predict(0.5)))
