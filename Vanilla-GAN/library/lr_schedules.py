import tensorflow as tf
class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self,initial_learning_rate,decay_factor,minimum_learning_rate,name=None):
        super(ExponentialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.minimum_learning_rate=minimum_learning_rate
        self.decay_factor = decay_factor
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or "ExponentialDecay") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate,
                                                         name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            
            minimum_learning_rate=tf.cast(self.minimum_learning_rate,dtype)
            decay_factor = tf.cast(self.decay_factor, dtype)
            t = tf.cast(step, dtype)
            
            #curr_lr= init_learning_rate / (decay_factor ** t)
            curr_lr=tf.divide(initial_learning_rate,tf.pow(decay_factor,t))
            
            return tf.clip_by_value(curr_lr,minimum_learning_rate,initial_learning_rate,name='clip_at_min')

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "decay_factor": self.decay_factor,
            "minimum_learning_rate":self.minimum_learning_rate,
            "name": self.name
        }