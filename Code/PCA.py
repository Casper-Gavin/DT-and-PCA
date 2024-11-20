import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

class MyPCA:
    def __init__(self):
        self.pca_results = {}
        self.read_csv_and_split_data()

    def read_csv_and_split_data(self):
        self.data = pd.read_csv('wine.csv')

        X = self.data.drop('Type', axis=1)  # 3 types of alcohol
        y = self.data['Type'] - 1 # 1-3 to 0-2
        
        # standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


    def pca_fit(self):
        pca = PCA()
        pca.fit(self.X_train)

        # find number of features needed
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        # given percentages
        percentages = [0.99, 0.80, 0.45]
        
        # Find number of components for each threshold
        for percentage in percentages:
            num_components = np.argmax(cumulative_variance >= percentage) + 1  # +1 to get the count
            self.pca_results[percentage] = {
                'num_components': num_components,
                'transformed_data': pca.transform(self.X_train)[:, :num_components]
            }

        # cumulative variance graph
        # plt.figure(figsize=(10, 5))
        # plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
        # plt.xlabel('Number of Components')
        # plt.ylabel('Cumulative Variance')
        # plt.title('Explained Cumulative Variance Against the Number of Principal Components')
        # plt.grid(True)
        # plt.show()

        # principle component graph
        # pca_2d = PCA(n_components=2)
        # X_train_pca_2d = pca_2d.fit_transform(self.X_train)
        # plt.figure(figsize=(10, 7))
        # scatter = plt.scatter(X_train_pca_2d[:, 0], X_train_pca_2d[:, 1], c=self.y_train, cmap='viridis', alpha=0.7)
        # plt.colorbar(scatter, label='Alochol Type')
        # plt.xlabel("PC 1")
        # plt.ylabel("PC 2")
        # plt.title("2D Plot of Strongest Principle Componenets (PCs) for Wine")
        # plt.show()  

    def train_mlp(self):
        # edit for specific percentage wanted
        num_components = self.pca_results[0.45]['num_components']
        X_train_transformed = self.pca_results[0.45]['transformed_data']
        
        # change test data for pca - fit
        pca = PCA(n_components=num_components)
        pca.fit(self.X_train)  # Fit PCA on the training data
        X_test_transformed = pca.transform(self.X_test)  # Transform the test data

        # Convert to PyTorch tensors
        X_train_t = torch.tensor(X_train_transformed, dtype=torch.float32)
        y_train_t = torch.tensor(self.y_train.values, dtype=torch.long)  # Use long for classification
        X_test_t = torch.tensor(X_test_transformed, dtype=torch.float32)
        y_test_t = torch.tensor(self.y_test.values, dtype=torch.long)

        # Create DataLoader
        train_data = TensorDataset(X_train_t, y_train_t)
        test_data = TensorDataset(X_test_t, y_test_t)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

        # Define the MLP model - created a small class
        class MLP(nn.Module):
            def __init__(self, input_size, num_classes):
                super(MLP, self).__init__()
                self.fc = nn.Linear(input_size, num_classes)

            # baisc forward pass for model: 2 relus, 1 hidden layer
            def forward(self, x):
                x = self.fc(x)  # No activation function for a single-layer perceptron
                return x
        
        input_size = X_train_transformed.shape[1]  # features number after pca
        num_classes = len(np.unique(self.y_train))   # unique classifications = 3 types of alcohol

        mlp_model = MLP(input_size, num_classes)

        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()  # For multi-class classification
        optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)

        # Training the model
        mlp_model.train()
        for epoch in range(100):  # You can adjust the number of epochs
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()  # Zero the gradients
                outputs = mlp_model(batch_X)  # Forward pass
                loss = criterion(outputs, batch_y)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

        # save model weights, do 99, 80, and 45
        torch.save(mlp_model.state_dict(), 'weights_45.pt')

        # Evaluate the model
        mlp_model.eval()  # Set the model to evaluation mode
        all_preds = []
        all_labels = []
        with torch.no_grad():  # No need to compute gradients during evaluation
            for batch_X, batch_y in test_loader:
                outputs = mlp_model(batch_X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(batch_y.numpy())

        # Calculate accuracy
        accuracy = accuracy_score(all_labels, all_preds)

        return accuracy
    
if __name__ == '__main__':
    pca = MyPCA()
    pca.pca_fit()
    acc = pca.train_mlp()
    print("Accuracy for 99%: " + str(acc))