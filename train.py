import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

data = np.loadtxt( "D:/VS Code/coursera/projects/titanic-predictor/data/data2.txt", delimiter=',')
X1_train = data[:, 0]
X2_train = data[:, 1]
X3_train = data[:, 2]
y_train = data[:, 3]

X_train = np.column_stack((X1_train, X2_train, X3_train))
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient_reg(X, y, w, b, lambda_=1):
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
    dj_dw += (lambda_ / m) * w
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, gradient_function, alpha, num_iters, lambda_):
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
                        compute_gradient_reg, alpha, iterations, lambda_)

# save model parts
joblib.dump((w, b), "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved as model.pkl & scaler.pkl")
