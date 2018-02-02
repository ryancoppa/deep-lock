from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy

dataset_zeros = numpy.zeros(shape=(1, 10))
dataset_ones = numpy.ones(shape=(1, 10))

dataset = numpy.append(dataset_zeros, dataset_ones, axis=1)

X = dataset.T
Y = dataset.T

model = Sequential()

model.add(Dense(1, input_dim=1))
model.add(Dense(10))
model.add(Dense(100))
model.add(Dense(1000))
model.add(Dense(10000))
model.add(Dense(1000))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=150, batch_size=10)

print("Predition for zero:")
predict_zero = numpy.zeros(shape=(1, 1))
print(model.predict_classes(predict_zero))

print("Predition for one:")
predict_one = numpy.ones(shape=(1, 1))
print(model.predict_classes(predict_one))
