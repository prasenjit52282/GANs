import tensorflow as tf

class Generator(tf.keras.Model):
    def __init__(self,z_dim):
        super(Generator,self).__init__()

        #layers
        self.lr1=tf.keras.layers.Dense(7*7*256, use_bias=False, input_shape=(z_dim,))
        self.bn1=tf.keras.layers.BatchNormalization()
        self.act1=tf.keras.layers.LeakyReLU()

        self.reshape=tf.keras.layers.Reshape((7, 7, 256))

        self.lr2=tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.bn2=tf.keras.layers.BatchNormalization()
        self.act2=tf.keras.layers.LeakyReLU()

        self.lr3=tf.keras.layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.bn3=tf.keras.layers.BatchNormalization()
        self.act3=tf.keras.layers.LeakyReLU()

        self.lr4=tf.keras.layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        
        #initilizing the model
        self.build(input_shape=(None,z_dim))
        self.call(tf.keras.layers.Input(shape=(z_dim,)))
        
    def call(self,x):
        net=self.lr1(x)
        net=self.bn1(net)
        net=self.act1(net)
        
        net=self.reshape(net)
        
        net=self.lr2(net)
        net=self.bn2(net)
        net=self.act2(net)
        
        net=self.lr3(net)
        net=self.bn3(net)
        net=self.act3(net)
        
        net=self.lr4(net) #tanh out
        
        return net
    
    
class Discriminator(tf.keras.Model):
    def __init__(self,in_shape):
        super().__init__()
        #layers
        self.lr1=tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                         input_shape=in_shape)
        self.act1=tf.keras.layers.LeakyReLU()
        self.dp1=tf.keras.layers.Dropout(0.3)

        self.lr2=tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.act2=tf.keras.layers.LeakyReLU()
        self.dp2=tf.keras.layers.Dropout(0.3)

        self.flatten=tf.keras.layers.Flatten()
        self.lr3=tf.keras.layers.Dense(1)#,activation="sigmoid")
       
        
        #initilizing the model
        self.build(input_shape=(None,*in_shape))
        self.call(tf.keras.layers.Input(shape=(*in_shape,)))
        
    def call(self,x):
        net=self.lr1(x)
        net=self.act1(net)
        net=self.dp1(net)
        
        net=self.lr2(net)
        net=self.act2(net)
        net=self.dp2(net)
        
        net=self.flatten(net)
        net=self.lr3(net) #sigmoid out not
        return net