from tensorflow import keras
model = keras.models.load_model('my_model')
print(model.summary())
# from keras.models import Model
# from keras import backend as K
# import numpy as np
#
# inp = model.input                                           # input placeholder
# outputs = [layer.output for layer in model.layers]          # all layer outputs
# functors = [K.function([inp, K.learning_phase()], [out]) for out in outputs]    # evaluation functions
#
# # Testing
# test = np.random.random(input_shape)[np.newaxis,...]
# layer_outs = [func([test, 1.]) for func in functors]
# print(layer_outs)