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

train_y = np.ravel(train_y)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def initialize_params(dim):
    w = np.zeros((dim,1))
    b = 0
    return w,b

def cost_compute(X,y,w,b,lamba = 1):
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X)+b)       
    cost = -(np.dot(y,np.log(A).T) + np.dot((1-y),np.log(1-A).T))/m  + (lamba/m)*(np.sum(np.power(w,2)))                               
    dw =  X.dot((A-y).T)/m + lamba*w
    db = np.sum(A - y)/m
   # cost = np.squeeze(cost)
    grads = {
            "dW" : dw,
            "db" : db
            }
    return cost,grads

def optimize(X,y,w,b,num_iterations,learning_rate = 0.005):
    costs = []
    for i in range(num_iterations):
        cost,grads = cost_compute(X,y,w,b)
        dW = grads["dW"]
        db = grads["db"]
        if i%100 == 0:
            costs.append(cost)
        w = w - learning_rate*dW
        b = b - learning_rate*db
    params = {
            "w":w,
            "b":b
            }
    return params,costs

def predict(X,params):
    w = params["w"]
    b = params["b"]
    A = sigmoid(np.dot(w.T,X)+b)
    my_prediction = A > 0.5
    return my_prediction

w,b = initialize_params(num_x*num_x*3)
params,costs = optimize(train_x,train_y,w,b,5000)
my_predic_test = predict(test_x,params)
my_predic_train = predict(train_x,params)

print("test accuracy: {} %".format(100 - np.mean(np.abs(my_predic_test - test_y)) * 100))
print("train accuracy: {} %".format(100 - np.mean(np.abs(my_predic_train - train_y)) * 100))

costs = np.squeeze(costs)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =0.005")
plt.show()


def check(image,params):
    my_image = np.array(plt.imread(image))
    my_image = tra.resize(my_image,(num_x,num_x)).reshape((1,num_x*num_x*3)).T
    prediction = predict(my_image,params)
    print("y = " + str(np.squeeze(prediction)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(prediction)),].decode("utf-8") +  "\" picture.")
    

check("images.jpg",params)








