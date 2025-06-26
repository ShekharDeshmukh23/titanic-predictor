from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import *
import os

app = Flask(__name__, static_folder='static')

# Load and preprocess data
data = np.loadtxt("data/data2.txt", delimiter=',')
X1_train = data[:, 0]  # age
X2_train = data[:, 1]  # gender (0/1)
X3_train = data[:, 2]  # ticket class
y_train = data[:, 3]   # survived

X_train = np.column_stack((X1_train, X2_train, X3_train))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, w, b):
    m, n = X.shape
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i], w) + b
        f_wb_i = sigmoid(z_i)
        cost += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    return cost / m

def compute_cost_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    cost_without_reg = compute_cost(X, y, w, b)
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost_without_reg + reg_cost

def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros_like(w)
    dj_db = 0.0
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        f_wb = sigmoid(z_wb)
        dj_db += f_wb - y[i]
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * X[i][j]
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw

def compute_gradient_reg(X, y, w, b, lambda_=1):
    m, n = X.shape
    dj_db, dj_dw = compute_gradient(X, y, w, b)
    dj_dw += (lambda_ / m) * w
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_):
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b, lambda_)
        w -= alpha * dj_dw
        b -= alpha * dj_db
    return w, b

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(3) - 0.5)
initial_b = -8
iterations = 10000
alpha = 0.001
lambda_ = 1.0

w, b = gradient_descent(X_train, y_train, initial_w, initial_b,
                        compute_cost_reg, compute_gradient_reg,
                        alpha, iterations, lambda_)

def predict_proba(X, w, b):
    m, n = X.shape
    probs = np.zeros(m)
    for i in range(m):
        z_wb = np.dot(X[i], w) + b
        probs[i] = sigmoid(z_wb)
    return probs

@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        X1 = float(data.get("age"))
        X2 = int(data.get("gender"))
        X3 = int(data.get("ticket_class"))

        X = np.array([[X1, X2, X3]])
        X_scaled = scaler.transform(X)
        proba = predict_proba(X_scaled, w, b)

        return jsonify({
            "probability": round(proba[0] * 100, 2),
            "prediction": "SURVIVE" if proba[0] >= 0.5 else "NOT SURVIVE"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
