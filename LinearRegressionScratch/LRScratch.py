import numpy as np  

class LinearRegression:
    def __init__(self, X, y, theta=None, lr=0.00005, converge=0.000001, max_iter=10000):
        # initialize model parameters and validate input shapes
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Number of samples in X and y must be the same")
        if theta is not None and X.shape[1] != theta.shape[0]:
            raise ValueError("Dimension of theta must match number of features in X")

        # set instance variables
        self.X = X
        self.y = y
        self.theta = theta if theta is not None else np.zeros(X.shape[1])
        self.lr = lr
        self.converge = converge
        self.max_iter = max_iter

    def h(self):
        # hypothesis function: calculate predictions
        return np.dot(self.X, self.theta)

    def cost(self):
        # compute cost function (Mean Squared Error)
        m = len(self.X)
        preds = self.h()
        costfn = (1 / (2 * m)) * np.sum((preds - self.y) ** 2)
        return costfn

    def gradient(self):
        # calculate gradient of the cost function
        m = len(self.X)
        preds = self.h()
        gr = (1 / m) * np.matmul(self.X.T, (preds - self.y))
        return gr

    def gradient_descent(self):
        # perform gradient descent to minimize cost function
        costfn_list = []
        for i in range(self.max_iter):
            gr = self.gradient()
            self.theta -= gr * self.lr
            costfn = self.cost()
            costfn_list.append(costfn)

            # check for convergence
            if len(costfn_list) > 1:
                if abs(costfn_list[-1] - costfn_list[-2]) < self.converge:
                    break
        return self.theta, costfn_list

    def predict(self, X_new):
        # predict values for new input data
        if not isinstance(X_new, np.ndarray):
            raise TypeError("X_new must be a numpy array")
        if X_new.shape[1] != self.theta.shape[0]:
            raise ValueError("Number of features in X_new must match the number of features in X")
        return np.dot(X_new, self.theta)
