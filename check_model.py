from tensorflow import keras
model = keras.models.load_model('my_model')
print(model.summary())
