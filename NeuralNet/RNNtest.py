import numpy as np

from sklearn.metrics import accuracy_score
from tensorflow import keras
#from keras.datasets import reuters
#from keras.preprocessing.sequence import pad_sequences
#from keras.utils import to_categorical
#from keras.models import Sequential
#from keras.layers import Dense, SimpleRNN, Activation
#from keras import optimizers
#from keras.wrappers.scikit_learn import KerasClassifier

# parameters for data load
num_words = 30000
maxlen = 50
test_split = 0.3

(X_train, y_train), (X_test, y_test) = keras.datasets.reuters.load_data(num_words=num_words, maxlen=maxlen, test_split=test_split)

# pad the sequences with zeros
# padding parameter is set to 'post' => 0's are appended to end of sequences
X_train = keras.preprocessing.sequence.pad_sequences(X_train, padding='post')
X_test = keras.preprocessing.sequence.pad_sequences(X_test, padding='post')

# reshape articles into column vectors
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

y_data = np.concatenate((y_train, y_test))
y_data = keras.utils.to_categorical(y_data)
y_train = y_data[:1395]
y_test = y_data[1395:]


def vanilla_rnn():
    model = keras.models.Sequential()
    model.add(keras.layers.SimpleRNN(50, input_shape=(49, 1), return_sequences=False))
    model.add(keras.layers.Dense(46))
    model.add(keras.layers.Activation('softmax'))

#    adam = keras.optimizers.Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


model = vanilla_rnn()     #KerasClassifier(build_fn=vanilla_rnn, epochs=200, batch_size=50, verbose=1)
model.fit(X_train, y_train, epochs=200, batch_size=50, verbose=1,
          validation_data=(X_test, y_test))


y_pred = model.predict(X_test)
y_test_ = np.argmax(y_test, axis=1)

print(accuracy_score(y_pred, y_test_))