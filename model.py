from layers import GraphConvolution, GraphConvolutionSparse, InnerDecoder, Dense
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS


class Model(object):

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class CAN(Model):

    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(CAN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
                                              
        self.hidden2 = Dense(input_dim=self.n_samples,
                                              output_dim=FLAGS.hidden1,
                                              act=tf.nn.tanh,
                                              sparse_inputs=True,
                                              dropout=self.dropout)(tf.sparse_transpose(self.inputs))
        self.z_u_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_u_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)
        self.z_a_mean = Dense(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       act=lambda x: x,
                                       dropout=self.dropout)(self.hidden2)

        self.z_a_log_std = Dense(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          act=lambda x: x,
                                          dropout=self.dropout)(self.hidden2)

        self.z_u = self.z_u_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_u_log_std)
        self.z_a = self.z_a_mean + tf.random_normal([self.input_dim, FLAGS.hidden2]) * tf.exp(self.z_a_log_std)

        self.reconstructions = InnerDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)((self.z_u, self.z_a))
