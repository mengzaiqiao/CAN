import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS

class OptimizerCAN(object):

    def __init__(self, preds, labels, model, num_nodes, num_features, pos_weight_u, norm_u, pos_weight_a, norm_a):
        preds_sub_u, preds_sub_a = preds
        labels_sub_u, labels_sub_a = labels
        self.cost_u = norm_u * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_u, targets=labels_sub_u, pos_weight=pos_weight_u))
        self.cost_a = norm_a * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub_a, targets=labels_sub_a, pos_weight=pos_weight_a))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost_u + self.cost_a
        self.kl_u = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_u_log_std - tf.square(model.z_u_mean) - 
                                                                   tf.square(tf.exp(model.z_u_log_std)), 1))
        self.kl_a = (0.5 / num_features) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_a_log_std - tf.square(model.z_a_mean) - 
                                                                   tf.square(tf.exp(model.z_a_log_std)), 1))
        self.kl = self.kl_u + self.kl_a
        
        self.cost = self.log_lik - self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction_u = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_u), 0.5), tf.int32),
                                           tf.cast(labels_sub_u, tf.int32))
        self.correct_prediction_a = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub_a), 0.5), tf.int32),
                                           tf.cast(labels_sub_a, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction_u, tf.float32)) + tf.reduce_mean(tf.cast(self.correct_prediction_a, tf.float32))
