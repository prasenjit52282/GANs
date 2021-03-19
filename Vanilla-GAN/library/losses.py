import tensorflow as tf

#D- is for Discriminator
#G is for Generator

def discriminator_loss(D,G,x,z):
    L_to_Maximize=tf.reduce_mean(tf.math.log(D(x))+tf.math.log(1-D(G(z))))
    L_to_Minimize=-L_to_Maximize # maximize L means minimize -L
    return L_to_Minimize


def generator_loss(D,G,z):
    L_to_Minimize=tf.reduce_mean(tf.math.log(1-D(G(z))))
    return L_to_Minimize