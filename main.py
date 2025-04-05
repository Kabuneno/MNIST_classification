import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train_train = X_train.reshape(X_train.shape[0],-1)
X_test_test = X_test.reshape(X_test.shape[0],-1)

weights = np.random.rand(784,10) * 0.01
biases =  np.zeros((1,10))
def forward_pass(X):
  res = (X@ weights) + biases
  return res
def one_hot(Z):
  res_mas = np.zeros((Z.shape[0],10))
  for i in range(Z.shape[0]):
    res_mas[i,Z[i]] = 1
  return res_mas

def loss_function(y_pred,y_true):
  m = y_true.shape[0]
  los = -np.sum(y_true * np.log(y_pred+ 1e-10))/m
  return los

def soft_max(Z):
  Z_stable = Z - np.max(Z,axis = 1 , keepdims = True)
  res =  np.exp(Z_stable) /  np.sum(np.exp(Z_stable),axis =1 , keepdims = True)
  return res

def compute_gradients(X,y_pred,y_true):
  m = X.shape[0]
  dZ  = y_pred - y_true
  dW =  (X.T @ dZ )/ m
  db = np.sum(dZ,axis = 0 , keepdims = True)
  return dW , db

def update_values(Wd,bd,learning_rate = 0.001):
  global weights, biases
  weights-= Wd * learning_rate
  biases-= bd * learning_rate


def train(X,y,epochs = 100 , batch_size =  32):
  m = X.shape[0]
  for epoch in range(epochs):
    indixes = np.random.permutation(m)
    X_shufle  = X[indixes]
    # X_shufle  = X_shufle.reshape(X_shufle.shape[0],-1)
    y_shuffle = y[indixes]
    for i in range(0,m,batch_size):
      X_batch  = X_shufle[i:i+batch_size]
      y_batch  = y_shuffle[i:i+batch_size]
      y_on_hot = one_hot(y_batch)
      Z = soft_max(forward_pass(X_batch))
      loss = loss_function(Z,y_on_hot)
      dW, db = compute_gradients(X_batch,Z,y_on_hot)

      update_values(dW,db)
    if epoch % 10 == 0:
      print(loss)


train(X_train_train,y_train)




def get_pred(X):
  res = soft_max(forward_pass(X))
  res_res = res.tolist()
  res = res_res[0].index(max(res_res[0]))
  return res

get_pred(X_test_test[0])

for i in range (100,110):

  plt.imshow(X_test[i])
  plt.show()

  if get_pred(X_test_test[i]) == y_test[i]:
    print(f"looks like a {get_pred(X_test_test[i])}")
    print("✅")
  else:
    print(f"looks like a {get_pred(X_test_test[i])} but it's {y_test[i]}")
    print("🔴")
