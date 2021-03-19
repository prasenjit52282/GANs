import tensorflow as tf

#D- is for Discriminator
#G is for Generator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True) #no sigmoid is activated at D

@tf.function
def discriminator_loss(D,G,x,z):
    real_output=D(x,training=True)
    fake_output=D(G(z,training=True),training=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
#     L_to_Maximize=tf.reduce_mean(tf.math.log(D(x))+tf.math.log(1-D(G(z))))
#     L_to_Minimize=-L_to_Maximize # maximize L means minimize -L
#     return L_to_Minimize

@tf.function
def generator_loss(D,G,z):
    fake_output=D(G(z,training=True),training=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)
#     L_to_Minimize=tf.reduce_mean(tf.math.log(1-D(G(z))))
#     return L_to_Minimize
#     L_to_Maximize=tf.reduce_mean(tf.math.log(D(G(z)))) # for inner (0->1) cost goto (-inf ,0)
#     L_to_Minimize=-L_to_Maximize # maximize L means minimize -L
#     return L_to_Minimize