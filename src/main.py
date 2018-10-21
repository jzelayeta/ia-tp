import tensorflow as tf
from sys import argv
from .src.training import training
from .test.testData import testData
from .test.predictions import predictions
import multiprocessing

def run_tensorflow():
    training(tf,model,ds_entrenamiento)
    testData(tf,ds_test,model)
    predictions(tf,model)


ds_entrenamiento = argv[1]
ds_test = argv[2]


tf.enable_eager_execution()

model = tf.keras.Sequential([#jugar con estos numeros si los resultados no dan bien...
      tf.keras.layers.Dense(10, activation="relu", input_shape=(3,)),
      tf.keras.layers.Dense(10, activation="relu"),
      tf.keras.layers.Dense(33) #Cantidad de resutados
    ])

p = multiprocessing.Process(target=run_tensorflow)
p.start()
p.join()
