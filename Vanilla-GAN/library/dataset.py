import tensorflow as tf

def load_mnist():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")
    return (x_train/255).reshape(-1,784)