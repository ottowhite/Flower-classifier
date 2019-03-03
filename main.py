from sklearn import tree
import pandas as pd

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
training_data = iris_data.iloc[:int(9*len(iris_data)/10), :]  # 135 items in training data (9/10 of the data)
testing_data = iris_data.iloc[int(9*len(iris_data)/10):, :]  # 15 items in testing data (1/10 of the data)

# splitting the training data into features and labels
training_features = training_data.loc[:, ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']].values
training_labels = training_data.loc[:, 'Species'].values

# splitting the testing data into features and labels
testing_features = testing_data.loc[:, ['Sepal length', 'Sepal width', 'Petal length', 'Petal width']].values
testing_labels = testing_data.loc[:, 'Species'].values

# training the decision tree with the training set
decision_tree = tree.DecisionTreeClassifier()
decision_tree.fit(training_features, training_labels)

# testing the trained decision trees results against the reserved data
result = (decision_tree.predict(testing_features) == testing_labels).tolist()  # converting to list from numpy array

# calculating the percentage accuracy of the decision tree
correct = 0
for x in result:
    correct += 1 if result[x] else False


print('The decision tree correctly categorised ' + str(correct) + " of 15 flower species. ")

print(str(testing_features[0]) + " " + str(testing_labels[0]))