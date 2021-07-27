from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Dropout,Dense
from tensorflow_core.python.keras import regularizers
from tensorflow.keras.models import Sequential
import tensorflow as tf

# Load the diabetes dataset
from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
print(diabetes_dataset["DESCR"])
print(diabetes_dataset.keys()) #data,target,feature_names,data_filename,target_filename
data = diabetes_dataset["data"]
targets = diabetes_dataset["target"]
print(targets)
# Normalise the target data (this will make clearer training curves)
targets = (targets - targets.mean(axis =0 ))/targets.std()
print(targets)

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)
print(type(train_data))
print(train_targets)
print(train_data)
print(train_data.shape)
print(test_data.shape)
print(train_targets.shape)
print(test_targets.shape)


def get_regularised_model(wd, rate):#wd:weight decay, rate:dropout rate
    model = Sequential([
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu", input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(1)
    ])
    return model


class TrainingCallback(Callback):
    def on_train_begin(self, logs=None):
        print("starting training...")

    def on_epoch_begin(self, epoch, logs=None):
        print( "starting epoch %s" %(epoch))

    def on_train_batch_begin(self, batch, logs=None):
        print("starting batch %s" % (batch))

    def on_train_batch_end(self, batch, logs=None):
        print("finish batch %s" %(batch))

    def on_epoch_end(self, epoch, logs=None):
        print("finish epoch %s" %(epoch))

    def on_train_end(self, logs=None):
        print("finish train...")


class TestingCallback(Callback):
    def on_test_begin(self, logs=None):
        print("starting training...")

    def on_epoch_begin(self, epoch, logs=None):
        print( "starting epoch %s" %(epoch))

    def on_test_batch_begin(self, batch, logs=None):
        print("starting batch %s" % (batch))

    def on_test_batch_end(self, batch, logs=None):
        print("finish batch %s" %(batch))

    def on_epoch_end(self, epoch, logs=None):
        print("finish epoch %s" %(epoch))

    def on_test_end(self, logs=None):
        print("finish train...")

model = get_regularised_model(0.00002,0.3)
model.summary()
#opt = tf.keras.optimizers.Adam(learning_rate=0.005)
# acc = tf.kears.metrics.SparceCategoricalAccuracy()
# mae = tf.kears.metrics.MeanAbsoluteError()
model.compile(optimizer="adam",loss="mse",metrics=["mae"])
history = model.fit(train_data,train_targets,epochs=20,validation_split=0.15,batch_size=64,verbose=False,callbacks=TrainingCallback)
model.evaluate(test_data,test_targets,verbose=2,callbacks=TestingCallback)







from sklearn.datasets import load_diabetes
diabetes_dataset = load_diabetes()
print(diabetes_dataset["DESCR"]) #数据schema信息
# Save the input and target variables
print(diabetes_dataset.keys()) # data,target,feature_names,data_filename,target_filename
data = diabetes_dataset["data"]
targets = diabetes_dataset["target"] #0~300之间的变化数字，需要对数据进行归一化
# Normalise the target data (this will make clearer training curves)
targets = (targets - targets.mean(axis=0))/targets.std()
# Split the data into train and test sets
from sklearn.model_selection import train_test_split
train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)
print(train_data.shape) #(397,10)
print(test_data.shape) # (45,10)
print(train_targets.shape) #(397,)
print(test_targets.shape) # (45,)
# Build the feedfoward model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def get_regularised_model(wd, rate):#wd:weight decay, rate:dropout rate
    model = Sequential([
        Dense(128, kernel_regularizer=regularizers.l2(wd), activation="relu", input_shape=(train_data.shape[1],)),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(128, kernel_regularizer=regularizers.l2(wd),activation="relu"),
        Dropout(rate),
        Dense(1)
    ])
    return model

def get_model():
   model =  Sequential([
       Dense(128,activation="relu",input_shape=(train_data.shape[1],)), # train_data.shape[1]设定输入的特征个数10
       Dense(128,activation="relu"),
       Dense(128,activation="relu"),
       Dense(128,activation="relu"),
       Dense(128,activation="relu"),
       Dense(128,activation="relu"),
       Dense(1) # 输出层
   ])
   return model
unregularised_model = get_model()
unregularised_model.compile(optimizer="adam",loss="mse")

#EarlyStopping早停机制，如果在迭代时loss减少小于，则停止训练，避免资源的浪费
#patience参数，过大可能造成过拟合，Number of epochs with no improvement after which training will be stopped
unreg_history = unregularised_model.fit(train_data,train_targets,epochs=100,batch_size=64,verbose=False,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)],validation_split=0.15)
unregularised_model.evaluate(test_data,test_targets,verbose=2) # 未加patience参数前：loss 0.5242

regularizer_model = get_regularised_model(0.00008,0.2)
regularizer_model.compile(optimizer="adam",loss="mse")
reg_history = regularizer_model.fit(train_data,train_targets,epochs=100,batch_size=64,verbose=False,
                                    validation_split=0.15,callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])
regularizer_model.evaluate(test_data,test_targets,verbose=2) # 未加patience参数前 loss 0.5305,比非正则化模型更糟
# Plot the training and validation loss

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(12, 5))

fig.add_subplot(121)

plt.plot(unreg_history.history['loss'])
plt.plot(unreg_history.history['val_loss'])
plt.title('Unregularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

fig.add_subplot(122)

plt.plot(reg_history.history['loss'])
plt.plot(reg_history.history['val_loss'])
plt.title('Regularised model: loss vs. epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')

plt.show()