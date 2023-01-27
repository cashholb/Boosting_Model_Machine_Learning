# Cashton Holbert
# 12/4/22
# Assignment 4
# 142 Machine Learning
import numpy as np

'''
Helper function to return euclidean distance between two points in N dimensional space
'''
def getEucDistance(pointA, pointB):
    euc_distance = 0
    for i in range(len(pointA)):
        euc_distance += (pointA[i] - pointB[i]) ** 2
    euc_distance = euc_distance ** float(1/2)
    return euc_distance

class BoostingClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.

    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = BoostingClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

    """
    def __init__(self):
        # initialize the parameters here
        self.num_iterations = 5
        self.confidence_factor = []
        self.pos_centroids = []
        self.neg_centroids = []
        self.ensemble = []

    def fit(self, X, y):
        # split data into neg and pos classes according to their label
        neg_class = [] 
        pos_class = [] 
        for i in range(len(y)):
            if y[i] == 1:
                pos_class.append(X[i])
            else:
                neg_class.append(X[i])

        # according to boosting algorithm, assign weights
        self.weights = np.full(len(X), 1/len(X))

        # start the boosting algorithm loop
        for t in range(self.num_iterations):
            print("Iteration %i:" % (t+1))

            # create seperate arrays to track the negative class and positive class weights
            neg_weights = []
            pos_weights = []
            for i in range(len(y)):
                if y[i] == 1:
                    pos_weights.append(self.weights[i])
                else:
                    neg_weights.append(self.weights[i])

            # calculate weighted centroids by averaging the respective class's 
            nCent = []
            pCent = []
            for i in range(len(X[0])):
                nCent.append(float(0))
                pCent.append(float(0))

            for i in range(len(neg_class)):
                for j in range(len(X[0])):
                    nCent[j] += float(neg_class[i][j] * neg_weights[i])
            for i in range(len(X[0])):
                nCent[i] = nCent[i] / (sum(neg_weights))

            for i in range(len(pos_class)):
                for j in range(len(X[0])):
                    pCent[j] += float(pos_class[i][j] * pos_weights[i])
            for i in range(len(X[0])):
                pCent[i] = pCent[i] / (sum(pos_weights))

            self.neg_centroids.append(nCent)
            self.pos_centroids.append(pCent)

            # calculate error rate on training set
            error_rate = 0
            predictions = []
            for data in X:
                
                negDistance = getEucDistance(data, self.neg_centroids[0])
                posDistance = getEucDistance(data, self.pos_centroids[0])

                if negDistance < posDistance:
                    predictions.append(-1)
                else:
                    predictions.append(1)

            for i in range(len(predictions)):
                if predictions[i] != y[i]:
                    error_rate += self.weights[i]

            print("Error = %s" % error_rate)

            # determine confidence factor as in slides
            cf = 0.5 * np.log((1-error_rate)/error_rate)
            print("Alpha = %s" % round(cf,4))
            self.confidence_factor.append(cf)

            # calculate the weights according to the boosting algo
            weight_incr = 1 / (2 * error_rate)
            weight_decr = 1 / (2 * (1 - error_rate))

            for i in range(len(predictions)):
                if predictions[i] == y[i]:
                    self.weights[i] = self.weights[i] * weight_decr
                else:
                    self.weights[i] = self.weights[i] * weight_incr

            print("Factor to increase weights = %s" % round(weight_incr, 4))
            print("Factor to decrease weights = %s" % round(weight_decr, 4))
            print("")

            # if the model converges before t = num_iterations (as the error_rate >= 0.5) then break out of the boosting algo
            if error_rate >= 0.5:
                # Track where loop was broken
                self.num_iterations = t
                break
            
        # the model converged early, make the following arrays of the correct size for calculations
        if self.num_iterations != 5:
            self.confidence_factor = self.confidence_factor[:self.num_iterations]
            self.neg_centroids = self.neg_centroids[:self.num_iterations]
            self.pos_centroids = self.pos_centroids[:self.num_iterations]

        self.confidence_factor = np.array(self.confidence_factor)
        self.neg_centroids = np.array(self.neg_centroids)
        self.pos_centroids = np.array(self.pos_centroids)

        # dot product the boosting meta weight (confidence_factor) and the neg/pos centroids as according to boosting algo
        neg_ensemble = self.confidence_factor.dot(self.neg_centroids)
        pos_ensemble = self.confidence_factor.dot(self.pos_centroids)

        # sum the both neg and pos ensembles and divide by the sum of the confidence factors as boosting describes
        self.ensemble.append(sum(neg_ensemble) / sum(self.confidence_factor))
        self.ensemble.append(sum(pos_ensemble) / sum(self.confidence_factor))
        self.ensemble = np.array(self.ensemble)

        return self

    def predict(self, X):
        
        pred_label = []
        # get euc distances and calculate closest point to make prediction
        for data in X:
            distToNeg = getEucDistance(data, self.neg_centroids[0])
            distToPos = getEucDistance(data, self.pos_centroids[0])
            if distToNeg < distToPos:
                pred_label.append(-1)
            else:
                pred_label.append(1)

        pred_label = np.array(pred_label)

        return pred_label
