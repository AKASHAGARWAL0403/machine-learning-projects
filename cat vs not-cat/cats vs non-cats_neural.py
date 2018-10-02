import numpy as np
import matplotlib.pyplot as plt
import skimage.transform as tra
import h5py

train_dataset = h5py.File('train_catvnoncat.h5', "r")
train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels

test_dataset = h5py.File('test_catvnoncat.h5', "r")
test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

m = train_set_x_orig.shape[0]
num_x = train_set_x_orig.shape[1]

train_set_x_orig = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_orig = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_x = train_set_x_orig / 255
test_x = test_set_x_orig / 255

def sigmoid(z):
    A = 1/(1+np.exp(-z))
    return A

def relu(z):
    A = np.maximum(0,z)
    return A
    
def sig_grad(dA,Z):
    s = sigmoid(Z)
    return dA*s*(1-s)

def relu_grad(dA,Z):
    dZ = np.array(dA,copy = True)
    dZ[Z<=0] = 0
    return dZ

def initialize_params(layer_dims):
    np.random.seed(1)
    L = len(layer_dims)
    parameters = {}
    for l in range(1,L):
        print(l)
        parameters["W"+str(l)] = np.random.randn(layer_dims[l],layer_dims[l-1]) / np.sqrt(layer_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layer_dims[l],1))
    return parameters

def forward_propogate(A,W,b):
    m = A.shape[1]
    Z = np.dot(W,A) + b  
    cache = {
            "A": A,
            "W": W,
            "b": b,
            "Z": Z
            }
    return cache

def f_propogation(X,params):
    caches = []
    L = len(params) // 2
    A = X
    for i in range(1,L):
        W = params["W"+str(i)]
        b = params["b"+str(i)]
        res = forward_propogate(A,W,b)
        caches.append(res)
        A = relu(res["Z"])
    W = params["W"+str(L)]
    b = params["b"+str(L)]
    res = forward_propogate(A,W,b)
    caches.append(res)
    AL = sigmoid(res["Z"])
    return AL,caches

def cost(AL,Y,caches,lamb=10):
    m = AL.shape[1]
    L = len(caches)
    c = 0
    for i in range(L):
        W = caches[i]["W"]
        c += (lamb/(2*m))*(np.sum(np.power(W,2)))
    c += (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))  
    return c

def backward_propogation(dZ,cache,lamb = 10):
    A_prev = cache["A"]
    W = cache["W"]
    m = A_prev.shape[1]
    #print("alasj"+str((lamb/m)*W))
    dW = (1./m * np.dot(dZ,A_prev.T)) + ((lamb/m)*W)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    grads = {
            "dW" : dW,
            "db" : db,
            "dA" : dA_prev
            }
    return grads
        
def b_propogation(AL,caches,params,Y):
    L = len(caches)
    grades = {}
    m = AL.shape[1]
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    dZ = sig_grad(dAL,current_cache["Z"])
    res = backward_propogation(dZ,current_cache)
    grades["dW"+str(L)] = res["dW"]
    grades["db"+str(L)] = res["db"]
    grades["dA"+str(L)] = res["dA"]
    for i in reversed(range(L-1)):
        dA = res["dA"]
        current_cache = caches[i]
        dZ = relu_grad(dA,current_cache["Z"])
        res = backward_propogation(dZ,current_cache)
        grades["dW"+str(i+1)] = res["dW"]
        grades["db"+str(i+1)] = res["db"]
        grades["dA"+str(i+1)] = res["dA"]
    return grades

def update_params(parameters,grads,learning_rate):
    L = len(parameters) // 2
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
    return parameters
    
def calc(X,Y,layer_dims,learning_rate = 0.008, num_iteration = 5000):
    np.random.seed(1)
    costs = []
    params = initialize_params(layer_dims)
    for i in range(num_iteration):
        AL,caches = f_propogation(X,params)
        c = cost(AL,Y,caches)
        if i%100 == 0:
            print(c)
            costs.append(c)
        grades = b_propogation(AL,caches,params,Y)
        params = update_params(params,grades,learning_rate)
    
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    return params

def predict(train_x, train_y,test_x,test_y):
    m = test_x.shape[1]
    p = np.zeros((1,m))
    test_res = []
    layer_dims = [12288, 20, 7, 5,1]
    for i in range(10,50):
        params_train = calc(train_x,train_y,layer_dims,0.008,i*100)
     #   predict(train_x,train_y,params_train)
     #   predict(test_x,test_y,params_train)
        probas, caches = f_propogation(test_x, params_train)
        for i in range(0, probas.shape[1]):
            if probas[0,i] > 0.5:
                p[0,i] = 1
            else:
                p[0,i] = 0
    
        print(p)
        test_res.append(np.sum((p == test_y)/m))
        print("Accuracy: "  + str(np.sum((p == test_y)/m)))
        
    return test_res

layer_dims = [12288, 20, 7, 5,1]
#params_train = calc(train_x,train_y,layer_dims)
test_res = predict(train_x,train_y,test_x,test_y)
plt.plot(np.squeeze(test_res))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate = 0.008")
plt.show()
#predict(test_x,test_y,params_train)

        
            












