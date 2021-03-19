import tensorflow as tf

def load_mnist():
    (x_train, y_train), (x_test, y_test)=tf.keras.datasets.mnist.load_data(path="mnist.npz")
    return ((x_train- 127.5) / 127.5).reshape(-1,28,28,1)