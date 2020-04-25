#!/usr/bin/env python

"""
Author: Ritvik Kapila
"""

# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


# In[2]:


# df = pd.DataFrame(columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])

# # Using readlines() to read data from text file 
# file1 = open('data.txt', 'r') 
# Lines = file1.readlines() 

# for i in list(range(0,len(Lines),2)):
#     newline = Lines[i].strip() + " " + Lines[i+1].strip() + "\n"
#     dftemp = pd.DataFrame([newline.split()], columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV'])
#     df = df.append(dftemp)

# df = df.reset_index()
# df = df.drop('index', axis = 1)
# df

df = pd.read_csv('data.csv')
df = (df-df.min())/(df.max()-df.min())


# In[3]:


y = np.array(df['MEDV'])
x = np.array(df.drop('MEDV', axis = 1))
# print(x,"\n","\n","\n",y)
y


# In[4]:


def MSE(y_pred, y_test):
    if(len(y_pred) == len(y_test)):
        return np.sum((y_pred-y_test)**2/len(y_test))
    else :
        print("Unequal lenghts of outputs")


# In[5]:


def kernel(a1, a2, kernel_type, gamma):    
    if(kernel_type == 'linear'):
        return np.matmul(a1, a2.T)
    elif(kernel_type == 'gaussian' or kernel_type == 'rbf'):
        K = np.zeros((a1.shape[0], a2.shape[0]))
        for i, xi in enumerate(a1):
            for j, xj in enumerate(a2):
                K[i,j] = np.exp(-gamma * np.linalg.norm(xi - xj) ** 2)
        return K
    elif(kernel_type == 'poly'):
#         return (a1@a2.T+1)**gamma
        K = np.zeros((a1.shape[0], a2.shape[0]))
        for i, xi in enumerate(a1):
            for j, xj in enumerate(a2):
                K[i,j] = (gamma*xi@xj+1)**2
        return K        


# # Epsilon SVR
# 

# In[6]:


m = x.shape[0]
y = y.reshape(m,1)

def svr(x_train, y_train, x_test, y_test, C, epsilon, kernel_type, gamma):
    m = x_train.shape[0]
    
    A = np.ones((1,m))
    A = np.concatenate((A,-A), axis = 1)
#     print(str(A) + "     Shape A : " + str(A.shape))

    b = np.zeros((1,1))
#     print(str(b) + "     Shape b : " + str(b.shape))
    
    G = np.identity(2*m)
    G = np.concatenate((G,-G), axis = 0)
#     print(str(G) + "     Shape G : " + str(G.shape))

    h = np.zeros((2*m,1))
    h = np.concatenate((h+C,h), axis = 0)
#     print(str(h) + "     Shape h : " + str(h.shape))
    
    q = np.ones((1,m))
    q = np.concatenate((q*epsilon-y_train.T,q*epsilon+y_train.T), axis = 1)
    q = q.T
#     print(str(q) + "     Shape q : " + str(q.shape))
    
    P11 = kernel(x_train, x_train, kernel_type, gamma)
    P12 = -P11
    P1 = np.concatenate((P11,P12), axis = 1)
    P2 = np.concatenate((P12,P11), axis = 1)
    P = np.concatenate((P1,P2), axis = 0)
#     print(str(P) + "     Shape P : " + str(P.shape))
    
    #Converting into cvxopt format
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    sol = solvers.qp(P, q, G, h, A, b)
    
    
    alpha = np.array(sol['x'])
    bias = np.array(sol['y'])
    a1 = alpha[:m][:]
    a2 = alpha[m:][:]
    w = np.matmul((a1-a2).T,x_train)     # Only for linear kernel as of now, make function phi
    
    f = np.matmul((a1-a2).T, kernel(x_train, x_test, kernel_type, gamma)) + bias
    return f, w, bias
    


# # RH SVR

# In[7]:


def RHsvr(x_train, y_train, x_test, y_test, D, epsilon, kernel_type, gamma):
    m = x_train.shape[0]
    
    A11 = np.ones((1,m))
    A12 = np.zeros((1,m))
    A1 = np.concatenate((A11,A12), axis = 1)
    A2 = np.concatenate((A12,A11), axis = 1)
    A = np.concatenate((A1,A2), axis = 0)
    
    b = np.ones((2,1))
    
    G = np.identity(2*m)
    G = np.concatenate((G,-G), axis = 0)
#     print(str(G) + "     Shape G : " + str(G.shape))

    h = np.zeros((2*m,1))
    h = np.concatenate((h+D,h), axis = 0)
#     print(str(h) + "     Shape h : " + str(h.shape))

    q = np.eye(m)
    q = 2 * epsilon * ((y_train.T)@np.concatenate((q,-q), axis = 1))
    q = q.T
    
    P11 = kernel(x_train, x_train, kernel_type, gamma) + y_train@(y_train.T)
    P12 = -P11
    P1 = np.concatenate((P11,P12), axis = 1)
    P2 = np.concatenate((P12,P11), axis = 1)
    P = np.concatenate((P1,P2), axis = 0)


    #Converting into cvxopt format
    P = matrix(P)
    q = matrix(q)
    G = matrix(G)
    h = matrix(h)
    A = matrix(A)
    b = matrix(b)

    sol = solvers.qp(P, q, G, h, A, b)
    
    
    alpha = np.array(sol['x'])
    
    u = alpha[:m][:]
    v = alpha[m:][:]
#     w = np.matmul((a1-a2).T,x_train)     # Only for linear kernel as of now, make function phi
    
    delta = ((u-v).T)@y_train + 2*epsilon
    bias = ((((u-v).T)@kernel(x_train, x_train, kernel_type, gamma))@(u+v))/(2*delta) + ((u+v).T)@y_train/2
    
    print(u,v)
    
    f = np.matmul((v-u).T, kernel(x_train, x_test, kernel_type, gamma))/delta + bias
    return f, bias


# In[8]:


y_pred, w, b = svr(x, y, x, y, 50.004999999999995, 0.01, 'rbf', 0.001)
y_pred.shape


# # SVR using sklearn

# In[9]:


from sklearn import svm

# reg = svm.SVR(kernel = 'poly', C = 0.5075, gamma = 0.575, degree = 2, epsilon = 0.01).fit(x, y.ravel())
reg = svm.SVR(kernel = 'rbf', gamma = 0.001, C = 50.004999999999995, epsilon = 0.01).fit(x, y.ravel())
# reg = svm.SVR(kernel = 'linear', C = 0.1, epsilon = 0.01).fit(x, y.ravel())
y1_pred = reg.predict(x)


# In[10]:


m = 506
xx = np.array(list(range(0,m)))
xx = xx.reshape(m,1)
plt.plot(xx[0:100], y[0:100], label = "Original")
plt.plot(xx[0:100], y_pred.reshape(m,1)[0:100], label = "SVR Prediction")
plt.plot(xx[0:100], y1_pred.reshape(m,1)[0:100], label = "sklearn Library")
plt.xlabel('Sample')
plt.ylabel('MEDV')
plt.legend()
plt.savefig('svr_lin_predictions_100_2.png')
plt.show()


# # Cross Validation

# In[11]:


def cross_val(x, y, C, epsilon, kernel_type, gamma, ita):
    m = len(y)
    errors = []
#     errors1 = []
    weights = [0]*ita
    bias = [0]*ita
    length_test_set = 0
#     x_train, y_train, x_test, y_test
    
    for i in range(ita):
        if(i == ita - 1):
            x_test = x[int(m/ita*(ita - 1)) :]
            y_test = y[int(m/ita*(ita - 1)) :]
            x_train = x[: int(m/ita*(ita - 1))]
            y_train = y[: int(m/ita*(ita - 1))]
            length_test_set = m - int((ita-1)*m/ita)

        else :
            x_test = x[int(m/ita*i) : int(m/ita*(i + 1))]
            y_test = y[int(m/ita*i) : int(m/ita*(i + 1))]
            x_train = np.concatenate((x[: int(m/ita*i)], x[int(m/ita*(i + 1)) :]), axis = 0)
            y_train = np.concatenate((y[: int(m/ita*i)], y[int(m/ita*(i + 1)) :]), axis = 0)
            length_test_set = int(m/ita)
            
        y_pred, w, b = svr(x_train, y_train, x_test, y_test, C, epsilon, kernel_type, gamma)
        
        y_pred = y_pred.reshape(length_test_set,1)
#         y1_pred = y_pred.reshape(length_test_set,1)
        y_test = y_test.reshape(length_test_set,1)
#         print("Shape of y_pred: " + str(y_pred.shape))
#         print("Shape of y_test: " + str(y_test.shape))
#         print(length_test_set)
        
        errors = np.append(errors, [MSE(y_pred,y_test)])
#         errors1 = np.append(errors1, [MSE(y1_pred,y_test)])
#         weights[i] = np.array([w])
#         bias[i] = b
    
#     for i in range(ita):
#         if(i==0):
#             w_eff = weights[0]
#             bias_eff = bias[0]
#         else :
#             w_eff = (w_eff + weights[i])
#             bias_eff = (bias_eff + bias[i])
#     w_eff = w_eff/ita 
#     bias_eff = bias_eff/ita
    return errors


# In[12]:


def cross_val_rh(x, y, C, epsilon, kernel_type, gamma, ita):
    m = len(y)
    errors = []
#     errors1 = []
    weights = [0]*ita
    bias = [0]*ita
    length_test_set = 0
#     x_train, y_train, x_test, y_test
    
    for i in range(ita):
        if(i == ita - 1):
            x_test = x[int(m/ita*(ita - 1)) :]
            y_test = y[int(m/ita*(ita - 1)) :]
            x_train = x[: int(m/ita*(ita - 1))]
            y_train = y[: int(m/ita*(ita - 1))]
            length_test_set = m - int((ita-1)*m/ita)

        else :
            x_test = x[int(m/ita*i) : int(m/ita*(i + 1))]
            y_test = y[int(m/ita*i) : int(m/ita*(i + 1))]
            x_train = np.concatenate((x[: int(m/ita*i)], x[int(m/ita*(i + 1)) :]), axis = 0)
            y_train = np.concatenate((y[: int(m/ita*i)], y[int(m/ita*(i + 1)) :]), axis = 0)
            length_test_set = int(m/ita)
            
        y_pred, b = RHsvr(x_train, y_train, x_test, y_test, C, epsilon, kernel_type, gamma)
        
        y_pred = y_pred.reshape(length_test_set,1)
#         y1_pred = y_pred.reshape(length_test_set,1)
        y_test = y_test.reshape(length_test_set,1)
#         print("Shape of y_pred: " + str(y_pred.shape))
#         print("Shape of y_test: " + str(y_test.shape))
#         print(length_test_set)
        
        errors = np.append(errors, [MSE(y_pred,y_test)])
#         errors1 = np.append(errors1, [MSE(y1_pred,y_test)])
#         weights[i] = np.array([w])
#         bias[i] = b
    
#     for i in range(ita):
#         if(i==0):
#             w_eff = weights[0]
#             bias_eff = bias[0]
#         else :
#             w_eff = (w_eff + weights[i])
#             bias_eff = (bias_eff + bias[i])
#     w_eff = w_eff/ita 
#     bias_eff = bias_eff/ita
    return errors


# In[13]:


def cross_val_lib(x, y, C, epsilon, kernel_type, gamma, ita, degree):
    m = len(y)
#     errors = []
    errors1 = []
    weights = [0]*ita
    bias = [0]*ita
    length_test_set = 0
#     x_train, y_train, x_test, y_test
    
    for i in range(ita):
        if(i == ita - 1):
            x_test = x[int(m/ita*(ita - 1)) :]
            y_test = y[int(m/ita*(ita - 1)) :]
            x_train = x[: int(m/ita*(ita - 1))]
            y_train = y[: int(m/ita*(ita - 1))]
            length_test_set = m - int((ita-1)*m/ita)

        else :
            x_test = x[int(m/ita*i) : int(m/ita*(i + 1))]
            y_test = y[int(m/ita*i) : int(m/ita*(i + 1))]
            x_train = np.concatenate((x[: int(m/ita*i)], x[int(m/ita*(i + 1)) :]), axis = 0)
            y_train = np.concatenate((y[: int(m/ita*i)], y[int(m/ita*(i + 1)) :]), axis = 0)
            length_test_set = int(m/ita)
                 
        reg = svm.SVR(kernel = kernel_type, C = C, gamma = gamma, degree = degree, epsilon = epsilon).fit(x_train, y_train.ravel())
            
        y1_pred = reg.predict(x_test)    
#         y_pred = y_pred.reshape(length_test_set,1)
        y1_pred = y1_pred.reshape(length_test_set,1)
        y_test = y_test.reshape(length_test_set,1)
#         print("Shape of y_pred: " + str(y_pred.shape))
#         print("Shape of y_test: " + str(y_test.shape))
#         print(length_test_set)
        
#         errors = np.append(errors, [MSE(y_pred,y_test)])
        errors1 = np.append(errors1, [MSE(y1_pred,y_test)])
#         weights[i] = np.array([w])
#         bias[i] = b
    
#     for i in range(ita):
#         if(i==0):
#             w_eff = weights[0]
#             bias_eff = bias[0]
#         else :
#             w_eff = (w_eff + weights[i])
#             bias_eff = (bias_eff + bias[i])
#     w_eff = w_eff/ita 
#     bias_eff = bias_eff/ita
    return errors1 


# In[14]:


err_model = cross_val_rh(x, y, 200, 0.01, 'poly', 1, 5)


# In[15]:


print(err_model)


# In[24]:


a = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.20]
b1 = [0] * len(a)
b2 = [0] * len(a)
b3 = [0] * len(a)
for i in range(len(a)):
    errors1 = cross_val(x, y, 0.1, 0.01, 'rbf', a[i], 5)
    errors2 = cross_val_lib(x, y, 0.1, 0.01, 'rbf', a[i], 5, 2)
    errors3 = cross_val_rh(x, y, 0.1, 0.01, 'rbf', a[i], 5)
    b1[i] = errors1.sum()/len(errors1)
    b2[i] = errors2.sum()/len(errors2)
    b3[i] = errors3.sum()/len(errors3)
    
    
print(b1)    
print(b2)
print(b3)


# In[25]:


print("a : " + str(a))
print("b1 : " + str(b1))
print("b2 : " + str(b2))
print("b3 : " + str(b3))

# plt.plot(a, b1, label = "SVR Predicion")
plt.plot(a, b1, linestyle='--', marker='o', color='b', label = "Epsilon SVR")
# plt.plot(a, b2, label = "sklearn Library")
plt.plot(a, b2, linestyle='--', marker='o', color='r', label = "sklearn Library")
plt.plot(a, b3, linestyle='--', marker='o', color='g', label = "RH SVR")
plt.title('Variation of MSE with Gamma')
plt.xlabel('Values of Gamma')
plt.ylabel('Mean Square Error')
plt.legend()
plt.savefig('rhsvr_poly_gamma.png')
plt.show()


# In[22]:


a = np.array([1, 10, 20, 30, 40, 50, 60, 70 ,80, 90, 100])/10
b1 = [0] * len(a)
b2 = [0] * len(a)
b3 = [0] * len(a)
for i in range(len(a)):
    errors1 = cross_val(x, y, a[i], 0.01, 'rbf', 0.01, 5)
    errors2 = cross_val_lib(x, y, a[i], 0.01, 'rbf', 0.01, 5, 2)
    errors3 = cross_val_rh(x, y, a[i], 0.01, 'rbf', 0.01, 5)
    b1[i] = errors1.sum()/len(errors1)
    b2[i] = errors2.sum()/len(errors2)
    b3[i] = errors3.sum()/len(errors3)
    
    
print(b1)    
print(b2)
print(b3)


# In[23]:


print("a : " + str(a))
print("b1 : " + str(b1))
print("b2 : " + str(b2))
print("b3 : " + str(b3))

# plt.plot(a, b1, label = "SVR Predicion")
plt.plot(a, b1, linestyle='--', marker='o', color='b', label = "Epsilon SVR")
# plt.plot(a, b2, label = "sklearn Library")
plt.plot(a, b2, linestyle='--', marker='o', color='r', label = "sklearn Library")
plt.plot(a, b3, linestyle='--', marker='o', color='g', label = "RH SVR")
plt.title('Variation of MSE with C')
plt.xlabel('Values of C')
plt.ylabel('Mean Square Error')
plt.legend()
plt.savefig('rhsvr_rbf_C.png')
plt.show()


# In[18]:


# y_pred, b = RHsvr(x, y, x, y, 1, 0.1, 'linear', 0.01)
y_pred2, b = RHsvr(x[:406], y[:406], x[406:], y[406:], 1, 0.1, 'linear', 0.01)


# In[114]:


from sklearn import svm

# reg = svm.SVR(kernel = 'poly', C = 100, gamma='auto', degree=2, epsilon = 1, coef0 = 1).fit(x, y.ravel())
# reg = svm.SVR(kernel = 'rbf', gamma = 1, C = 1000, epsilon = 1).fit(x, y.ravel())
reg = svm.SVR(kernel = 'linear', C = 0.1, epsilon = 0.001).fit(x, y.ravel())
y1_pred = reg.predict(x)


# In[117]:


m = len(y)
xx = np.array(list(range(0,m)))
xx = xx.reshape(m,1)
plt.plot(xx[0:506], y.reshape(m,1)[0:506], label = "Original")
plt.plot(xx[0:506], y_pred.reshape(m,1)[0:506], label = "Predicted")
# plt.plot(xx[0:100], y1_pred.reshape(m,1)[0:100], label = "Library")
plt.xlabel('x - axis')
plt.ylabel('MEDV')
plt.legend()
plt.show()


# In[116]:


from sklearn.metrics import r2_score

y = y.reshape(m,1)
y_pred = y_pred.reshape(m,1)
r2_score(y, y_pred, sample_weight=None, multioutput='uniform_average')


# In[16]:


np.array([1,2,3])@np.array([3])


# In[ ]:




