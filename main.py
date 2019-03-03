import math
import pandas as pd


def find_distance(a, b):
    distance_squared = 0
    for x in range(0, len(a)):
        distance_squared += (a[x] - b[x])**2
    distance =  math.sqrt(distance_squared)
    return distance


class KNearestNeighbors:

    def __init__(self):
        self.labels = [0, 1, 2]
        self.predictions = []

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)

        return predictions

    def closest(self, feature):
        # initially it is assumed that the closes distance is the first item
        closest_distance = find_distance(feature, self.X_train[0])
        closest_index = 0

        # iterating over the training data and testing distance, the first item has already been evaluated
        for i in range(1, len(self.X_train)):
            distance = find_distance(self.X_train[i], feature)
            if distance < closest_distance:
                closest_distance = distance
                closest_index = i

            return self.y_train[closest_index]  # returns the label of the closes neighbour


iris_data = pd.read_csv("iris_flower_data.csv")

flower_species = ["I. setosa", "I. versicolor", "I. virginica"]
column_names = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Species']

# Setosa is 0
# Versicolor is 1
# Virginica is 2
iris_data.loc[iris_data["Species"].isin(["I. setosa"]), "Species"] = 0
iris_data.loc[iris_data["Species"].isin(["I. versicolor"]), "Species"] = 1
iris_data.loc[iris_data["Species"].isin(["I. virginica"]), "Species"] = 2

# randomising the order of the set
iris_data = iris_data.sample(frac=1).reset_index(drop=True)

# splitting the data
training_data = iris_data.iloc[:int(len(iris_data)/2), :]  # 75 items in training data (9/10 of the data)
testing_data = iris_data.iloc[int(len(iris_data)/2):, :]  # 75 items in testing data (1/10 of the data)

# splitting the training data into features and labels
training_features = training_data.loc[:, ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']].values
training_labels = training_data.loc[:, 'Species'].values

# splitting the testing data into features and labels
testing_features = testing_data.loc[:, ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']].values
testing_labels = testing_data.loc[:, 'Species'].values

# training the decision tree with the training set
decision_tree = KNearestNeighbors()
decision_tree.fit(training_features, training_labels)

# testing the trained decision trees results against the reserved data
result = (decision_tree.predict(testing_features) == testing_labels).tolist()  # converting to list from numpy array

# calculating the percentage accuracy of the decision tree
correct = 0
for x in result:
    correct += 1 if result[x] else False

print('The k nearest neigbors algorithm correctly categorised ' + str(correct) + " of 75 flower species. ")
correct = correct / 75 * 100
print('The algorithm performed with ' + str(correct) + '% accuracy.' )