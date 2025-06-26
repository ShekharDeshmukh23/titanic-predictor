import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from sklearn.preprocessing import StandardScaler

data = np.loadtxt("D:/VS Code/coursera/projects/data/data2.txt", delimiter=',')  


X1_train = data[:, 0]  
X2_train = data[:, 1]  
X3_train = data[:, 2]  
y_train  = data[:, 3]  

X_train = np.column_stack((X1_train, X2_train, X3_train)) 

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

print("First five elements in X1_train are:\n", X1_train[:5])            #age
print("Type of X1_train:",type(X1_train))
print("First five elements in X1_train are:\n", X2_train[:5])            #gender(male=0,female=1)
print("Type of X2_train:",type(X2_train))
print("First five elements in X1_train are:\n", X3_train[:5])            #ticketClass
print("Type of X3_train:",type(X3_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))

print ('The shape of X_train is: ' + str(X1_train.shape))
print ('The shape of X_train is: ' + str(X2_train.shape))
print ('The shape of X_train is: ' + str(X3_train.shape))
print ('The shape of y_train is: ' + str(y_train.shape))
print ('We have m = %d training examples' % (len(y_train)))


def sigmoid(z):
   
    g = 1/(1+np.exp(-z))
    
    return g

def compute_cost(X, y, w, b, *argv):
  
    m, n = X.shape
    
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    total_cost = cost / m

    return total_cost

def compute_cost_reg(X, y, w, b, lambda_ = 1):
  
    m, n = X.shape
    
    cost_without_reg = compute_cost(X, y, w, b) 
  
    reg_cost = 0.
    
    for j in range(n):
        reg_cost += (w[j]**2)                                        
    reg_cost = (lambda_/(2*m)) * reg_cost   
   
    total_cost = cost_without_reg + reg_cost

    return total_cost

def compute_gradient(X, y, w, b, *argv): 
   
    m, n = X.shape
    dj_dw = np.zeros(w.shape)
    dj_db = 0.

    for i in range(m):
        z_wb = 0.
        for j in range(n): 
            z_wb += X[i][j] * w[j]
        z_wb += b
        f_wb = sigmoid(z_wb)
        
        dj_db_i = f_wb - y[i]
        dj_db += dj_db_i
        
        for j in range(n):
            dj_dw[j] += (f_wb - y[i]) * X[i][j]
            
    dj_dw = dj_dw / m
    dj_db = dj_db / m
  
    return dj_db, dj_dw

def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
   
    m, n = X.shape
    
    dj_db, dj_dw = compute_gradient(X, y, w, b)

    for j in range(n):
        dj_dw[j] = dj_dw[j] + (lambda_/m) * w[j]

    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
  
    m = len(X)
    
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        if i<100000:     
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history 

np.random.seed(1)
initial_w = 0.01 * (np.random.rand(3) - 0.5)
initial_b = -8

iterations = 10000
alpha = 0.001

lambda_ = 1.0
w,b, J_history,_ = gradient_descent(X_train, y_train, initial_w, initial_b, 
                                   compute_cost_reg, compute_gradient_reg, 
                                   alpha, iterations, lambda_)


def predict(X, w, b): 
 
    m, n = X.shape   
    p = np.zeros(m)
  
    for i in range(m):   
        z_wb = 0.
       
        for j in range(n): 
          
            z_wb += X[i][j] * w[j]
     
        z_wb += b
      
        f_wb = sigmoid(z_wb)

        p[i] = 1 if f_wb >= 0.5 else 0
   
    return p

def predict_proba(X, w, b): 
    m, n = X.shape
    probs = np.zeros(m)

    for i in range(m):   
        z_wb = np.dot(X[i], w) + b
        probs[i] = sigmoid(z_wb)

    return probs


p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))

X1 = float(input("Enter your age: "))
print("The entered number is:", X1)
X2 = int(input("Enter your gender(male:0 , female:1): "))
print("The entered number is:", X2)
X3 = int(input("Enter ticket class(1,2,3): "))
print("The entered number is:", X3)

X = np.array([[X1, X2, X3]])
X_scaled = scaler.transform(X)
proba = predict_proba(X_scaled, w, b)

print(f"Predicted Survival Probability: {proba[0] * 100:.2f}%")

if proba[0] >= 0.5:
    print("🎉 You would likely SURVIVE the Titanic.")
else:
    print("💀 You would likely NOT survive the Titanic.")








