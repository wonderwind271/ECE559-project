import numpy as np
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

def data_gen(line: tuple, num_negative: int, num_positive: int, margin: float, seed: int):
    np.random.seed(seed)
    dim = len(line) - 1 

    def sample_point(label):
        """ Generate a single point that satisfies the margin constraint. """
        while True:
            x = np.random.uniform(-10, 10, dim)  
            value = line[0] + np.dot(line[1:], x) 
            
            if label == -1 and value <= -margin:
                return x
            if label == 1 and value >= margin:
                return x

    X_neg = np.array([sample_point(-1) for _ in range(num_negative)])
    X_pos = np.array([sample_point(1) for _ in range(num_positive)])

    X = np.vstack((X_neg, X_pos))
    y = np.array([-1] * num_negative + [1] * num_positive)
    indices = np.arange(X.shape[0])  # Create index array
    np.random.shuffle(indices)  # Shuffle indices
    X, y = X[indices], y[indices]

    return X, y


def save_dataset(X, y, filename="dataset"):
    np.save(f"{filename}_X.npy", X)
    np.save(f"{filename}_y.npy", y)
    print(f"Saved {filename}_X.npy and {filename}_y.npy")


def load_dataset(filename="dataset"):
    X = np.load(f"{filename}_X.npy")
    y = np.load(f"{filename}_y.npy")
    print(f"Loaded {filename}_X.npy and {filename}_y.npy")
    return X, y


def train_svm_get_hyperplane(X, y):
    """
    Train a hard-margin SVM and return the hyperplane coefficients (b0, b1, ..., bn).
    
    Parameters:
        X (np.ndarray): Feature matrix of shape (N, d), where d is the number of features.
        y (np.ndarray): Labels (-1 or +1), shape (N,).
    
    Returns:
        model, tuple: (b0, b1, ..., bn), where:
            - b0 is the bias term (intercept),
            - b1, ..., bn are the weight vector components.
    """
    # Train a hard-margin SVM (C is set to a large value)
    model = SVC(kernel="linear", C=1e6)
    model.fit(X, y)
    
    # Extract coefficients
    w = model.coef_[0]  # Weight vector (b1, ..., bn)
    b0 = model.intercept_[0]  # Bias term
    
    # Create the tuple (b0, b1, ..., bn)
    hyperplane = (b0, *w)
    
    print(f"Trained SVM hyperplane: {hyperplane}")
    return model, hyperplane


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Linear transformation

    def forward(self, x):
        return self.linear(x)  # No sigmoid needed (BCEWithLogitsLoss handles it)
    

def train_logistic_regression_gd(X, y, epochs=100, lr=0.01, model=None):
    """
    Train logistic regression using PyTorch with gradient descent.

    Parameters:
        X (np.ndarray): Feature matrix of shape (N, d).
        y (np.ndarray): Labels (-1 or +1), shape (N,).
        epochs (int): Number of training epochs.
        lr (float): Learning rate for optimization.

    Returns:
        model
        tuple: (b0, b1, ..., bn) where:
            - b0 is the bias term,
            - b1, ..., bn are the weight vector components.
    """
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape for BCE loss
    # y_tensor = (y_tensor+1)/2
    print(y_tensor.shape)

    # Initialize model
    if model is None:
        model = LogisticRegressionModel(X.shape[1])
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss (logistic loss)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_tensor)  # Forward pass
        loss = criterion(outputs, y_tensor)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Print progress
        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # Extract trained parameters
    w = model.linear.weight.detach().numpy().flatten()  # Weights
    b0 = model.linear.bias.detach().numpy().item()  # Bias term

    # Return as a tuple (b0, b1, ..., bn)
    return model, (b0, *w)


def train_logistic_regression_sgd(X, y, steps=1000, batch_size=32, lr=0.001):
    """
    Train logistic regression using Stochastic Gradient Descent (SGD).

    Parameters:
        X (np.ndarray): Feature matrix of shape (N, d).
        y (np.ndarray): Labels (-1 or +1), shape (N,).
        steps (int): Total number of optimization steps.
        batch_size (int): Mini-batch size for SGD.
        lr (float): Learning rate for optimization.

    Returns:
        model
        tuple: (b0, b1, ..., bn) where:
            - b0 is the bias term,
            - b1, ..., bn are the weight vector components.
    """
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Reshape for BCE loss

    # Create DataLoader for mini-batch training
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = LogisticRegressionModel(X.shape[1])
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss (logistic loss)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # Training loop using steps instead of epochs
    step = 0
    while step < steps:
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()  # Clear gradients
            outputs = model(batch_X)  # Forward pass
            loss = criterion(outputs, batch_y)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            step += 1
            if step % (steps // 10) == 0 or step == 1:
                print(f"Step [{step}/{steps}], Loss: {loss.item():.4f}")
            
            if step >= steps:
                break  # Stop after reaching the step count

    # Extract trained parameters
    w = model.linear.weight.detach().numpy().flatten()  # Weights
    b0 = model.linear.bias.detach().numpy().item()  # Bias term

    # Return as a tuple (b0, b1, ..., bn)
    return model, (b0, *w)

def plot_line_from_coeffs(xs, X, y, path=None):
    for x in xs:
        a0, a1, a2 = x

        # Domain for x1
        x1 = np.linspace(-10, 10, 400)

        # Avoid division by zero
        if a2 == 0:
            # Vertical line: a1 * x1 = -a0 ⇒ x1 = -a0 / a1
            x_val = -a0 / a1
            plt.axvline(x=x_val, color='red', label=f"x1 = {-a0/a1:.2f}")
        else:
            # Solve for x2
            x2 = -(a0 + a1 * x1) / a2
            plt.plot(x1, x2, label=f"{a0:.1f} + {a1:.1f}x1 + {a2:.1f}x2 = 0")

    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.legend()
    plt.title(f"Plot of the line {x}")
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolors="k")
    if path is None:
        plt.show()
    else:
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.clf()

