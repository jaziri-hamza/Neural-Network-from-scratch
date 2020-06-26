import numpy as np



from sklearn.datasets import load_breast_cancer # binaryclass
data = load_breast_cancer()
inputs = data.data
targets = data.target.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(inputs, targets, test_size=0.1)


# BINARY CLASSIFICATION
from NN import NN
from Layer import Layer
nn = NN(X_train, Y_train)
nn.add_layer( Layer(16, activation='relu') )
nn.add_layer( Layer(32, activation='sigmoid') )

# train
nn.fit(iteration=1, learning_rate=0.001)

import matplotlib.pyplot as plt
plt.plot(nn._costs)
plt.show()
pred  = nn.predict( X_train )

### MULTI CLASSIFICAITON
# from sklearn.datasets import load_iris # multiclass
# data = load_iris()
# inputs = data.data
# targets = data.target.reshape(-1,1)
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(inputs, targets, test_size=0.1)

# ### encode target labels
# from Encoder import Encoder
# enc = Encoder()
# Y_train_enc = enc.encode(Y_train)

# from NN import NN
# from Layer import Layer

# nn = NN(X_train, Y_train_enc)
# nn.add_layer( Layer(32, activation='sigmoid') )
# nn.add_layer( Layer(10, activation='sigmoid') )
# nn.fit()

# pred = nn.predict(X_train)
# pred = enc.decode(pred)


# from sklearn.datasets import load_boston

# data = load_boston()
# inputs = data.data
# targets = data.target.reshape(-1,1)

# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(inputs, targets, test_size=0.1)
# from NN import NN
# nn = NN(X_train, Y_train, output_activation='no_func')

# from Layer import Layer

# nn.add_layer( Layer(100, activation='relu') )
# nn.add_layer( Layer(5, activation='relu') )
# nn.add_layer( Layer(32, activation='relu') )

# nn.fit()

# import matplotlib.pyplot as plt

# plt.plot(nn._costs)
# plt.show()

# pred = nn.predict(X_train[0:30,:])
# print( Y_train[0:30,:])
# print( pred )

def accuary(Y_true, Y_pred):
    true = 0
    false = 0
    for i in range(Y_true.shape[0]):
        if Y_true[i,:] == Y_pred[i,:]:
            true +=1
        else:
            false +=1
    print("True", true/ Y_true.shape[0] )
    print("False", false / Y_true.shape[0])


accuary(Y_train, pred)
