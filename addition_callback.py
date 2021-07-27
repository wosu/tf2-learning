'''
https://colab.research.google.com/drive/1JC780dIqlK4Moip1jJjRWn6ZiXdNYEST#forceEdit=true&sandboxMode=true&scrollTo=3s7k9CAZX1SY
'''
import tensorflow as tf
print(tf.__version__)
from sklearn.datasets import  load_diabetes
from sklearn.model_selection import train_test_split

diabetes_dataset = load_diabetes()

# Save the input and target variables
data = diabetes_dataset['data']
targets = diabetes_dataset['target']
# Split the data set into training and test sets
train_data,test_data,train_targets,test_targets = train_test_split(data,targets,test_size=0.1)
#Let's also build a simple model to fit to the data with our callbacks.
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = tf.keras.Sequential([
    Dense(128,activation='relu',input_shape=(train_data.shape[1],)),
    Dense(64,activation='relu'),
    Dense(64,activation='relu'),
    Dense(1)
])
#compile the model
model.compile(optimizer="adam",loss="mse",metrics=["mse","mae"])


#Usage: tf.keras.callbacks.LearningRateScheduler(schedule, verbose=0)
# Define the learning rate schedule function
def lr_function(epoch,lr): #lr:learning rate
    if epoch % 2 ==0:
        return lr
    else:
        return lr + epoch/1000
#train the model
history = model.fit(train_data,train_targets,epochs=10,callbacks=
[tf.keras.callbacks.LearningRateScheduler(lr_function,verbose=1)],verbose=False)

#You can also use lambda functions to define your schedule given an epoch.
# Train the model with a difference schedule
history = model.fit(train_data,train_targets,epochs=10,callbacks=
[tf.keras.callbacks.LearningRateScheduler(lambda x:1/(3+5*x),verbose=1)])

#csv logger
#Usage tf.keras.callbacks.CSVLogger(filename, separator=',', append=False)