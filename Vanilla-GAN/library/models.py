import tensorflow as tf
import tensorflow_addons as tfa

class Generator(tf.keras.Model):
    def __init__(self,z_dim,out_dim):
        super().__init__()
        #kernel init
        self.w_init=tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)

        #layers
        self.linear1=tf.keras.layers.Dense(units=1200,kernel_initializer=self.w_init)
        self.relu1=tf.keras.layers.ReLU()
        
        self.linear2=tf.keras.layers.Dense(units=1200,kernel_initializer=self.w_init)
        self.relu2=tf.keras.layers.ReLU()
        
        self.linear3=tf.keras.layers.Dense(units=out_dim,kernel_initializer=self.w_init)
        
        #initilizing the model
        self.build(input_shape=(None,z_dim))
        self.call(tf.keras.layers.Input(shape=(z_dim,)))
        
    def call(self,x):
        net=self.linear1(x)
        net=self.relu1(net)
        
        net=self.linear2(net)
        net=self.relu2(net)
        
        net=tf.nn.sigmoid(self.linear3(net))
        return net
    
    
class Discriminator(tf.keras.Model):
    def __init__(self,in_dim):
        super().__init__()
        #kernal init
        self.w_init=tf.keras.initializers.RandomUniform(minval=-0.005, maxval=0.005)
        
        self.linear1=tf.keras.layers.Dense(units=1200,kernel_initializer=self.w_init)
        self.maxout1=tfa.layers.Maxout(num_units=240)
        
        self.linear2=tf.keras.layers.Dense(units=1200,kernel_initializer=self.w_init)
        self.maxout2=tfa.layers.Maxout(num_units=240)
        
        self.linear3=tf.keras.layers.Dense(units=1,kernel_initializer=self.w_init)
        
        #initilizing the model
        self.build(input_shape=(None,in_dim))
        self.call(tf.keras.layers.Input(shape=(in_dim,)))
        
    def call(self,x):
        net=self.linear1(x)
        net=self.maxout1(net)
        
        net=self.linear2(net)
        net=self.maxout2(net)
        
        net=tf.nn.sigmoid(self.linear3(net))
        return net