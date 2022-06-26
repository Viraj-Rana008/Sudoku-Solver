from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import SudokuNet

learning_rate = 1e-3
epochs = 10
batch = 128

print("[INFO] accessing MNIST...")
((trainX, trainY), (testX, testY)) = mnist.load_data()

# taking grayscale values
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# normalize data to [0 to 1]
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

# convert labels from integer to vectors
le = LabelBinarizer()
trainY = le.fit_transform(trainY)
testY = le.fit_transform(testY)


print("[INFO] compiling model...")
opt = Adam(lr=learning_rate)
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# train network
print("[INFO] training network...")
H = model.fit(trainX, trainY, 
              validation_data=(testX, testY), 
              batch_size=batch, 
              epochs=epochs,
              verbose=1)


print("[INFO] evaluating network...")
predictions = model.predict(testX)
print(classification_report(
    testY.argmax(axis=1),
    predictions.argmax(axis=1),
    target_names=[str(x) for x in le.classes_]
))

# save model
print("[INFO] serializing model...")
model.save('./SudokuSolver/model/myModel.h5', save_format="h5")
