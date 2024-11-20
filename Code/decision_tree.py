import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score


class MyDecisionTree:
    def __init__(self):
        self.read_csv_and_split_data()

    def read_csv_and_split_data(self):
        # Load the data
        self.data=pd.read_csv('diabetes.csv')

    def train_and_predict(self):
        # Train the decision tree
        train_acc = 0.0
        test_acc = 0.0

        X = self.data.drop(columns='Outcome')  # all columns except 'Outcome'
        y = self.data['Outcome']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
        
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)

        y_pred_train = dt_model.predict(X_train)
        y_pred_test = dt_model.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        print("Train accuracy score", train_acc)
        print("Test accuracy score", test_acc)

        # plt.figure(figsize=(30, 15), dpi=200)

        # plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['No Diabetes', 'Diabetes'], rounded=True, fontsize=2)

        # # Set a title and display the plot
        # plt.title("Diabetes Dataset Decision Tree")
        # plt.show()

        return train_acc, test_acc

if __name__ == '__main__':
    classifier = MyDecisionTree()
    classifier.train_and_predict()
    