import numpy as np
import pandas as pd
import pickle
import os

# Load and preprocess the dataset
dataset = pd.read_csv("data.csv")
dataset["Rain"] = dataset["Rain"].map({"rain": 1, "no rain": 0})

# Prepare training data
x_train = dataset[["Temperature (°C)", "Humidity (%)"]].values
y_train = dataset["Rain"].values

# Get data dimensions
q, n = x_train.shape


def sigmoid(z):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-z))


def cost_function(x, y, m, c):
    """Calculate logistic regression cost using cross-entropy loss"""
    z = np.dot(x, m) + c
    g = sigmoid(z)
    cost = -(y * np.log(g + 1e-9)) - (1 - y) * np.log(1 - g + 1e-9)

    return np.mean(cost)


def gradient_function(x, y, m, c):
    """Calculate gradients for weights and bias"""
    z = np.dot(x, m) + c
    g = sigmoid(z)
    error = g - y
    grad_m = np.dot(x.T, error) / q
    grad_c = np.sum(error) / q

    return grad_m, grad_c


def gradient_descent(x, y, lr, it):
    """Train the model using gradient descent optimization"""
    # Initialize parameters
    m = np.zeros(n)
    c = 0

    # Training loop
    for i in range(it + 1):
        grad_m, grad_c = gradient_function(x, y, m, c)

        # Update parameters
        m -= lr * grad_m
        c -= lr * grad_c

        # Print training progress
        if i % 1000 == 0:
            print(f"Iteration {i}: Loss = {cost_function(x, y, m, c):.6f}")

    return m, c


def predict(x, m, c):
    """Make predictions using trained model"""
    g = sigmoid(np.dot(x, m) + c)
    return g


# Model loading or training (if model is not present in the same directory as the model.py it will train a new model otherwise will load the model if present in same dit)
if os.path.exists("model.pkl"):
    print("Loading existing model...")
    with open("model.pkl", "rb") as f:
        m, c = pickle.load(f)
        print("Model loaded successfully")

else:
    print("No saved model found. Starting training...")

    # Default hyperparameters
    lr = 0.01
    it = 100000

    # Get user input for training parameters
    lr = float(input("Enter learning rate (default=0.01): ") or lr)
    it = int(input("Enter number of iterations (default=100000): ") or it)

    print(f"Training with learning rate: {lr}, iterations: {it}")

    # Train the model
    m, c = gradient_descent(x_train, y_train, lr, it)

    print(
        f"Training completed. Final Loss: {cost_function(x_train, y_train, m, c):.6f}"
    )

    # Save trained model
    with open("model.pkl", "wb") as f:
        pickle.dump((m, c), f)
        print("Model saved successfully")

# Prediction loop
print("\n--- Prediction Mode ---")
while True:
    q = input("Do you want to predict? (y/n): ").lower()
    if q == "n":
        print("Exiting program")
        break

    # Get input features from user input
    feature_cols = ["Temperature (°C)", "Humidity (%)"]
    user_input = []

    for col in feature_cols:
        val = input(f"Enter value for {col}: ")
        user_input.append(float(val))

    # Make prediction
    s = pd.DataFrame([user_input], columns=feature_cols)
    pred_prob = predict(s.values, m, c)
    pred_rain = "rain" if (pred_prob >= 0.5) else "no rain"

    # Display results
    print(f"Predicted Probability: {pred_prob[0]:.4f}")  #We used [0] so that the output doesnt show in a array and prints as a float(in this case)
    print(f"Rain Prediction: {pred_rain}")
