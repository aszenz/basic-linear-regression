import numpy as np
class LinearRegressor:
  """
  A simple linear regression algorithm
  """

  def __init__(self, no_of_features=2):
    """Initialize parameters to zeros"""
    # no of parameters will be equal to no of features
    self.parameters = np.zeros((no_of_features), dtype='float')
    self.no_of_features = no_of_features

  def _batch_gd_(self, X, y, learning_rate):
    no_of_examples = len(X)
    # current prediction for X using current model parameters for all examples
    y_predict = np.array([self.predict(x) for x in X])
    # update each parameter (weight) based on the derivative for all training examples
    for index in range(len(self.parameters)):
      derivative = sum([(y[i] - y_predict[i]) * X[i][index] for i in range(no_of_examples)])
      # LMS Rule - Least Mean Square Update Rule
      self.parameters[index] = self.parameters[index] + learning_rate * derivative
  
  def _iterative_gd_(self, X, y, learning_rate):
    no_of_examples = len(inputs)
    # stochastic gradient descent algorithm
    for i in range(0, no_of_examples):
      y_predict = self.predict(X[i])
      # update each parameter (weight) based on the derivative of one training example
      for index in range(len(self.parameters)):
        derivative = (y[i] - y_predict) * X[i][index]
        # LMS Rule -- Least Mean Square Update Rule
        self.parameters[index] = self.parameters[index] + learning_rate * derivative
  
  def predict(self, input_data):
    """Predict output given input"""
    # return the output
    return np.dot(self.parameters.transpose(), input_data)

  def fit(self, inputs, outputs, learning_rate=0.01, iterations=1000, gd_method='batch'):
    """Train the model as ordinary least squares regression model"""
    if gd_method == 'batch':
      for i in range(iterations):
        self._batch_gd_(inputs, outputs, learning_rate)
    elif gd_method == 'stochastic':
      for i in range(iterations):
        self._iterative_gd_(inputs, outputs, learning_rate)
    else:
      raise BaseException("Only batch and stochastic accepted values for gd_method")      
    print('Model parameters: %s' % self.parameters)

inputs = [[1, 2, 3], [3, 1, 3], [4, 4, 2], [1, 9, 3], [0, 3, 2]]
outputs = [2, 2.33, 3.33, 4.33, 1.66]
inputs = np.array(inputs)
outputs = np.array(outputs)
LR = LinearRegressor(no_of_features=3)
LR.fit(inputs, outputs, gd_method='batchs')
print('Output: %f' % LR.predict(np.array([10, 2, 5])))
LR.fit(inputs, outputs, learning_rate=0.001, iterations=2000, gd_method='stochastic')
print('Output: %f' % LR.predict(np.array([10, 2, 5])))

