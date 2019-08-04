#importing required modules

from sklearn.externals import joblib
from sklearn.datasets import fetch_mldata
import numpy as np
#for creating Neural Network  I am using  MLPClassifier from sklearn

from sklearn.neural_network.multilayer_perceptron import  MLPClassifier

#getting MNIST of size 70k images
dataset =fetch_mldata("MNIST original")
X = np.array(dataset.data)  #Our Features
y = np.array(dataset.target) #Our labels

X =  X.astype('float32') 

#splitting Dataset into Training and Testing dataset
#First 60k instances are for Training and last 10k are for testing
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]


#Normalizing Our Features in range 0 and 1
X_train = X_train /255
X_test = X_test /255

#creating Neural Network
# Neural Network has one hidden layer with 512 units
# Neural NetWork is of size 784-512-10

mlp = MLPClassifier(hidden_layer_sizes=(512), max_iter=500, verbose=True)

#fitting our model
mlp.fit(X_train, y_train,epoch=50)


print("Training set score: %f" % mlp.score(X_train, y_train)) #output : 0.99
print("Test set score: %f" % mlp.score(X_test, y_test))     #output :0.98

#saving our model
joblib.dump(mlp, "model.pkl")
