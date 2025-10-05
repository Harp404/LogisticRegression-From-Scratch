# 🤖 Logistic Regression From Scratch

A complete implementation of logistic regression algorithm built from scratch using Python, NumPy, and Pandas. This project demonstrates binary classification for rain prediction based on temperature and humidity data. ☔

## ✨ Features

- **🐍 Pure Python Implementation**: Built without using sklearn or other ML libraries
- **📊 Binary Classification**: Predicts rain/no rain based on weather conditions
- **💾 Model Persistence**: Saves and loads trained models using pickle
- **🎮 Interactive Prediction**: User-friendly interface for making predictions
- **📈 Gradient Descent Optimization**: Custom implementation of gradient descent algorithm

## 📋 Dataset

The model uses weather data with the following features:
- **🌡️ Temperature (°C)**: Temperature readings in Celsius
- **💧 Humidity (%)**: Humidity percentage values
- **☔ Rain**: Target variable (rain/no rain)

## 📁 Project Structure

```
LogisticRegression From Scratch/
├── model.py          # Main implementation file
├── data.csv          # Training dataset
├── model.pkl         # Saved trained model (created after first run)
├── requirements.txt  # Project dependencies
└── README.md         # Project documentation
```

## 🚀 Installation

1. Clone or download this project
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 🏃‍♂️ Running the Model

```bash
python model.py
```

### 🏋️‍♀️ First Run (Training)
- If no saved model exists, the program will train a new model
- You can specify custom learning rate and iterations, or use defaults:
  - Default learning rate: 0.01
  - Default iterations: 100,000

### ⚡ Subsequent Runs (Loading)
- The program automatically loads the saved model for faster startup
- Skip training and go directly to predictions

### 🔮 Making Predictions
- Enter temperature and humidity values when prompted
- The model will output:
  - Predicted probability (0-1 range)
  - Final prediction (rain/no rain)

## 🧮 Algorithm Details

### 🔧 Logistic Regression Components

1. **📊 Sigmoid Function**: Maps any real number to (0,1) range
2. **💰 Cost Function**: Uses logistic loss (cross-entropy)
3. **⬇️ Gradient Descent**: Optimizes model parameters iteratively
4. **🎯 Prediction**: Uses 0.5 threshold for binary classification

### 🧪 Mathematical Foundation

- **📐 Hypothesis**: h(x) = σ(θᵀx + b)
- **💲 Cost Function**: J(θ) = -1/m Σ[y·log(h(x)) + (1-y)·log(1-h(x))]
- **🔄 Gradient Updates**: θ := θ - α·∇J(θ)

## 🔑 Key Functions

- `sigmoid(z)`: Activation function
- `cost_function(x, y, m, c)`: Calculates logistic loss
- `gradient_function(x, y, m, c)`: Computes gradients
- `gradient_descent(x, y, lr, it)`: Trains the model
- `predict(x, m, c)`: Makes predictions on new data

## 📊 Model Performance

The model training displays:
- 📉 Loss reduction over iterations
- 🎯 Final loss value after training
- 📈 Training progress every 1000 iterations

## 🔧 Customization

You can modify:
- ⚡ Learning rate for different convergence speeds
- 🔁 Number of iterations for training duration
- 📊 Feature columns in the dataset
- 🎯 Prediction threshold (currently 0.5)

## 📋 Requirements

- 🐍 Python 3.7+
- 🔢 NumPy 1.21.0+
- 🐼 Pandas 1.3.0+

## 📝 Example Output

```
Model not found. Training new model...
Enter learning rate for training(default=0.01): 
Enter number of iterations you want model to train for(default=100000): 
iteration no:0 and loss is 0.6931471805599453
iteration no:1000 and loss is 0.4234567890123456
...
Final Loss: 0.1234567890123456
Model saved successfully!

Do You Want To Predict? (y/n): y
Enter value for Temperature (°C): 25
Enter value for Humidity (%): 80
Predicted Probability: 0.8234567890123456
Rain Prediction: rain
```

## 📄 License

This project is open source and available under the Apache-2.0 License 📄.
