from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential([
    Dense(64,activation='sigmoid',input_shape=(10,)),
    Dense(1)
])
model.compile(optimizer="sgd", loss=BinaryCrossentropy(from_logits=True))


#方式1：使用模型检查点call_back保存模型
checkpoint = ModelCheckpoint('my_model', save_weights_only=True)
model.fit(X_train,y_train,epochs=10, callbacks=[checkpoint])
#checkpoint 每个epoch的weights会被覆盖
#my_model.data-00000-of-00001
#my_model.index

#keras hdf5格式保存，只保存了权重，没有保存模型的结构，需要重建模型结构
checkpoint = ModelCheckpoint('keras_model.h5', save_weights_only=True)
model.fit(X_train,y_train,epochs=10, callbacks=[checkpoint])
#keras_model.h5

#加载以前保存的权重
model = Sequential([
    Dense(64,activation='sigmoid',input_shape=(10,)),
    Dense(1)
])
#第一种
model.load_weights('my_model')
#第二种
model.load_weights('keras_model.h5')

#使用save方法保存模型
model.compile(optimizer="sgd",loss="mse", metrics=["mae"])
early_stopping = EarlyStopping(monitor="val_mae",patience=2)
model.fit(X_train,y_train,validation_split=0.2,epochs=50,callbacks=[early_stopping])
model.sample_weights('my_model')





#model checkpoint
import tensorflow as tf
#CIFAR-10 image dataset, with total 60000 color images,with 10 labels(about animals)
#load data sets and rescale the pixel values
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

#use smaller subset to speed up
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:10000]
y_test = y_test[:10000]

#plot image demo
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,10,figsize=(10,1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshoe(x_train[i])


def get_test_accuracy(model,x_test,y_test):
    test_loss,test_acc = model.evaluate(x=x_test,y=y_test,verbose=0)
    print('accuracy : {acc:0.3f}'.format(acc=test_acc))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,   Flatten, Conv2D, MaxPooling2D

def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32,32,3), kernel_size=(3,3),activation='relu', name='conv_1'),
        Conv2D(filters=8,kernel_size=(3,3),activation='relu',name='conv_2'),
        MaxPooling2D(pool_size=(4,4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32,activation='relu',name='dense_1'),
        Dense(units=10,activation='softmax',name='dense_2')
    ])
    model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', metrics=['accuracy'])
    return
#train model with checkpoint
from tensorflow.keras.callbacks import ModelCheckpoint
checkpoint_path = 'model_checkpoints/checkpoint'
#每个epoch结束保存model时，都会覆盖上一次epoch的结果
checkpoint = ModelCheckpoint(filepath=checkpoint_path,frequency='epoch',save_weights_only=True,verbose=1)
model.fit(x_train,y_train,epochs=3,callbacks=[checkpoint])
get_test_accuracy(model,x_test,y_test) # accuracy 0.477
#create new instance of initialised model,accuracy around 10% again
model = get_new_model()
get_test_accuracy(model,x_test,y_test) # 0.081
#load weigts of the train checkpoint
model.load_weights(checkpoint_path)
get_test_accuracy(model,x_test,y_test) #accuracy 0.477


#Model saving criteria
#训练期间保存网络模型，并使用更灵活的选项
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

model = Sequential([
    Dense(16,activation='relu'),
    Dropout(0.3),
    Dense(3,activation='softmax')
])
model.compile(optimizer='rmsprop',loss='sparse_category_crossentropy',metrics=['acc','mse'])
#save_freq设置save的频次，默认值为epoch,1000表示样本数量，每1000个样本的权重，1000/16，表示‘
#每62或63此迭代保存一次,save_best_only参数会根据monitor设置的性能测量标准保存权重
#model参数告诉回调函数，我们是通过最大化还是最小化监视器的设置来衡量指标
#默认值为auto,会自动从设置的性能度量指标中，推测出往那个方向走（最大or最小）
#my_model.{epoch}.{batch}每次epoch和批次保留到对应的文件中
#{epoch}-{val_loss:.4f}将loss作为文件
# checkpoint = ModelCheckpoint('training_run_1/my_model.{epoch}.{batch}',save_weights_only=True,save_freq=1000,
#                              save_best_only=True,monitor='val_loss',model='max')
checkpoint = ModelCheckpoint('training_run_1/my_model.{epoch}-{val_loss:.4f}',save_weights_only=True,save_freq=1000,
                             save_best_only=True,monitor='val_loss',model='max')
model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=10,batch_size='16',callbacks=[checkpoint])



#实验
import tensorflow as tf
#CIFAR-10 image dataset, with total 60000 color images,with 10 labels(about animals)
#load data sets and rescale the pixel values
(x_train,y_train),(x_test,y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0

#use smaller subset to speed up
x_train = x_train[:10000]
y_train = y_train[:10000]
x_test = x_test[:10000]
y_test = y_test[:10000]

#plot image demo
import matplotlib.pyplot as plt
fig,ax = plt.subplots(1,10,figsize=(10,1))
for i in range(10):
    ax[i].set_axis_off()
    ax[i].imshoe(x_train[i])


def get_test_accuracy(model,x_test,y_test):
    test_loss,test_acc = model.evaluate(x=x_test,y=y_test,verbose=0)
    print('accuracy : {acc:0.3f}'.format(acc=test_acc))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,   Flatten, Conv2D, MaxPooling2D

def get_new_model():
    model = Sequential([
        Conv2D(filters=16, input_shape=(32,32,3), kernel_size=(3,3),activation='relu', name='conv_1'),
        Conv2D(filters=8,kernel_size=(3,3),activation='relu',name='conv_2'),
        MaxPooling2D(pool_size=(4,4), name='pool_1'),
        Flatten(name='flatten'),
        Dense(units=32,activation='relu',name='dense_1'),
        Dense(units=10,activation='softmax',name='dense_2')
    ])
    model.compile(optimizer='adam', loss='sparse_categorial_crossentropy', metrics=['accuracy'])
    return model
checkpoint_5000_path = 'model_checkpoints_5000/checkpint_{epoch:02d}_{batch:04d}'
checkpoint_5000 = ModelCheckpoint(filepath=checkpoint_5000_path,save_weights_only=True,
                                  save_freq=5000,verbose=1)
model = get_new_model()
#checkpoint_01_0499 checkpoint_02_0999 (500*batch_size=5000)
model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test),
          batch_size=10,callbacks=[checkpoint_5000])

#save_best_only
checkpoint_best_path = 'model_checkpoints_best/checkpoint'
checkpoint_best = ModelCheckpoint(filepath=checkpoint_best_path,save_weights_only=True,
                                  save_freq='epoch',monitor='val_accuracy',save_best_only=True,verbose=1)
history = model.fit(x_train,y_train,epochs=50,batch_size=10,validation_data=(x_test,y_test),
                    callbacks=[checkpoint_best],verbose=0)
#过拟合
import pandas as pd
pd.DataFrame(history,history)
#load model weights
new_model = get_new_model()
new_model.load_weights(checkpoint_best_path)
get_test_accuracy(new_model,x_test,y_test)

from tensorflow.keras.layers import BatchNormalization
#Saving the entire model
model = Sequential([
    Dense(16,activation='relu'),
    Dropout(0.3),
    Dense(3,activation='softmax'),
    BatchNormalization()
])
model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics=['acc','mae'])

#save_weights_only=False,不仅仅保存模型参数，还会保存模型结构
#my_model/assets/ tf图形所使用的文件位置
#my_model/saved_model.pb 表示的是模型的结构
#variables文件夹，包含模型保存的权重
#my_model/variables/variables.data-00000-of-00001
#my_model/variables/variables.index
checkpoint = ModelCheckpoint('my_model',save_weights_only=False)
model.fit(x_train,y_train,epochs=10,callbacks=[checkpoint])
#当指定保存的模型的格式时
#会仅仅生成my_model.h5文件，包含模型的权重，以及体系结构
checkpoint = ModelCheckpoint('my_model.h5',save_weights_only=False)
model.fit(x_train,y_train,epochs=10,callbacks=[checkpoint])
#model.save
model.save('my_model.h5')
#加载模型
from tensorflow.keras.models import load_model
new_model = load_model('my_model') #保存的模型的路径，会返回整个模型的架构，即使没有参数权重
new_model.summary()
new_model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=20,batch_size=16)
new_model.evaluate(x_test,y_test)
new_model.predict(x_samples)
