####This code trains a 3-hidden layer deep auto encoder in a greedy layerwise fashion; then transfers the encoder weights for initializing
####the corresponding 3-hidden layer MLP

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils import np_utils

input_units = 784
hidd1_units= 500      #number of units in the first hidden layer
output1_units = input_units
nb_classes = 10

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#AE_module1
ae_module1 = Sequential()
ae_module1.add(Dense(hidd1_units, input_dim=input_units, init='uniform', activation='relu'))
ae_module1.add(Dense(output1_units, activation='sigmoid'))
ae_module1.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
ae_module1.fit(X_train, X_train, nb_epoch=30, batch_size=100, shuffle=True)

ae_module1.save_weights('ae_module1.h5')
ae_module1.load_weights('ae_module1.h5')

g1=ae_module1.get_config()
h1=ae_module1.get_weights()

#Note here that enc is hidden layer and dec is output layer
enc1_w= h1[0]
enc1_bias_w= h1[1]
enc1_dec1_w= h1[2]
dec1_bias_w= h1[3]
#concatenate horizonatlly as array list module1 enc1 and enc1_bias weights
enc1_with_bias_w= [(enc1_w), (enc1_bias_w)]     #keep this for deep model fine-tuning
enc1_w_trans= enc1_w.transpose()    #transpose enc1 weights
#get shape for enc1 bias from input data size
#Note that by default (originally), input data has samples as rows and attributes as columns
[x1, y1] = np.shape(X_train.transpose())
#Here, activations computation follow my usual way; y= W(transpose)*x + bias weight*1s
enc1_bias_inp = np.ones((1,y1), dtype= np.int)
#compute the pre-activations of enc1, including bias contribution
enc1_pre_acts = np.matmul(enc1_w_trans, X_train.transpose()) + np.outer(enc1_bias_w, enc1_bias_inp)

#Pass enc1 pre-activations through activation function to obtain enc1 activations
#Here list the common activation functions implementation in numpy
#NOTE: use the same activation function used in the pre-training phase of enc1
#ReLU: y= np.maximum(x, 0, x); sigmoid: y= 1 / (1 + np.exp(-x)); tanh: y = np.tanh(x)
enc1_acts= np.maximum(enc1_pre_acts, 0, enc1_pre_acts)  #enc1 activations here

#############################Auto encoder module 2 goes here
#############################
#############################

#Transpose enc1 activation into Keras original format
ae_module2_input= enc1_acts.transpose()
#Set auto encoder 2 parameters
enc2_input_units = hidd1_units
hidd2_units= 400          #number of units in the second hidden layer
output2_units = enc2_input_units

#Define AE_module2 architecture
ae_module2 = Sequential()
ae_module2.add(Dense(hidd2_units, input_dim= enc2_input_units, init='uniform', activation='relu'))
ae_module2.add(Dense(output2_units, activation='sigmoid'))
ae_module2.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
ae_module2.fit(ae_module2_input, ae_module2_input, nb_epoch=30, batch_size=100, shuffle=True)

ae_module2.save_weights('ae_module2.h5')
ae_module2.load_weights('ae_module2.h5')

g2=ae_module2.get_config()
h2=ae_module2.get_weights()

#Note here that enc is hidden layer and dec is output layer
enc2_w= h2[0]
enc2_bias_w= h2[1]
enc2_dec2_w= h2[2]
dec2_bias_w= h2[3]

#concatenate horizonatlly as array list module1 enc and bias weights
enc2_with_bias_w= [(enc2_w), (enc2_bias_w)]     #keep this for deep model fine-tuning
enc2_w_trans= enc2_w.transpose()    #transpose enc1 weights
#get shape for enc2 bias from input data size
#Note that by default (originally), input data has samples as rows and attributes as columns
[x2, y2] = np.shape(ae_module2_input.transpose())
#Here, activations computation follow my usual way; y= W(transpose)*x + bias weight*1s
enc2_bias_inp = np.ones((1,y2), dtype= np.int)
#compute the pre-activations of enc1, including bias contribution
enc2_pre_acts = np.matmul(enc2_w_trans, ae_module2_input.transpose()) + np.outer(enc2_bias_w, enc2_bias_inp)

#Pass enc1 pre-activations through activation function to obtain enc1 activations
#Here list the common activation functions implementation in numpy
#NOTE: use the same activation function used in the pre-training phase of enc1
#ReLU: y= np.maximum(x, 0, x); sigmoid: y= 1 / (1 + np.exp(-x)); tanh: y = np.tanh(x)
enc2_acts= np.maximum(enc2_pre_acts, 0, enc2_pre_acts)  #enc1 activations here

#############################Auto encoder module 3 goes here
#############################
#############################

#Transpose enc3 activation into Keras original format
ae_module3_input= enc2_acts.transpose()
#Set auto encoder 2 parameters
enc3_input_units = hidd2_units
hidd3_units= 300              #number of units in the third hidden layer
output3_units = enc3_input_units

#Define AE_module2 architecture
ae_module3 = Sequential()
ae_module3.add(Dense(hidd3_units, input_dim= enc3_input_units, init='uniform', activation='relu'))
ae_module3.add(Dense(output3_units, activation='sigmoid'))
ae_module3.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
ae_module3.fit(ae_module3_input, ae_module3_input, nb_epoch=30, batch_size=100, shuffle=True)

ae_module3.save_weights('ae_module3.h5')
ae_module3.load_weights('ae_module3.h5')

g3=ae_module3.get_config()
h3=ae_module3.get_weights()

#Note here that enc is hidden layer and dec is output layer
enc3_w= h3[0]
enc3_bias_w= h3[1]
enc3_dec3_w= h3[2]
dec3_bias_w= h3[3]

#concatenate horizonatlly as array list module1 enc and bias weights
enc3_with_bias_w= [(enc3_w), (enc3_bias_w)]     #keep this for deep model fine-tuning

#Finetune(ft) model using encoder weights of trained auto encoder
ae_module_ft = Sequential()
module1_enc_ft_w= [(enc1_w), (enc1_bias_w)]

ae_module_ft.add(Dense(hidd1_units, input_dim=input_units, init='uniform', activation='relu'))
ae_module_ft.layers[0].set_weights(enc1_with_bias_w)

ae_module_ft.add(Dense(hidd2_units, activation='relu'))
ae_module_ft.layers[1].set_weights(enc2_with_bias_w)

ae_module_ft.add(Dense(hidd3_units, activation='relu'))
ae_module_ft.layers[2].set_weights(enc3_with_bias_w)

ae_module_ft.add(Dense(nb_classes, activation='softmax'))
ae_module_ft.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

ae_module_ft.fit(X_train, Y_train, nb_epoch=100, batch_size=100, shuffle=True, validation_data=(X_test, Y_test))

print('end of fine-tuning')
train_perf = ae_module_ft.evaluate(X_train, Y_train, verbose=0)
test_perf = ae_module_ft.evaluate(X_test, Y_test, verbose=0)
print('Train accuracy:', train_perf[1])
print('Test accuracy:', test_perf[1])

#Next, let's visualize ten of the input images and the reconstructed outputs from auto encoder module1.
decoded_imgs = ae_module1.predict(X_test)
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    #Convert test image
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    #How many items to display
    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
