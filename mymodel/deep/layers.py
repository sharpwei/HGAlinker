import tensorflow as tf
import numpy as np

from . import inits

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class MultiLayer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties    
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, edge_type=(), **kwargs):

        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolutionSparseMulti(MultiLayer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj_mats,
                 nonzero_feat, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparseMulti, self).__init__(**kwargs)
        self.dropout = dropout
        self.adj_mats = adj_mats
        self.act = act
        self.issparse = True
        self.nonzero_feat = nonzero_feat
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights' ] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights')

    def _call(self, inputs):
        outputs = []

        x = dropout_sparse(inputs, 1-self.dropout, self.nonzero_feat)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights' ])
        x = tf.sparse_tensor_dense_matmul(self.adj_mats, x)
        outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)
        return outputs


class GraphConvolutionMulti(MultiLayer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj_mats, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionMulti, self).__init__(**kwargs)
        self.adj_mats = adj_mats
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['weights'] = inits.weight_variable_glorot(
                    input_dim, output_dim, name='weights')

    def _call(self, inputs):
        outputs = []
        x = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj_mats, x)
        outputs.append(self.act(x))
        outputs = tf.add_n(outputs)
        outputs = tf.nn.l2_normalize(outputs, dim=1)
        return outputs


class DEDICOMDecoder(MultiLayer):
    """DEDICOM Tensor Factorization Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DEDICOMDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):
            self.vars['global_interaction'] = inits.weight_variable_glorot(
                input_dim, input_dim, name='global_interaction')
            tmp = inits.weight_variable_glorot(
                    input_dim, 1, name='local_variation')
            self.vars['local_variation'] = tf.reshape(tmp, [-1])

    def _call(self, inputs):

        outputs = []
        inputs_row = tf.nn.dropout(inputs, 1-self.dropout)
        inputs_col = tf.nn.dropout(inputs, 1-self.dropout)
        relation = tf.diag(self.vars['local_variation'])
        product1 = tf.matmul(inputs_row, relation)
        product2 = tf.matmul(product1, self.vars['global_interaction'])
        product3 = tf.matmul(product2, relation)
        rec = tf.matmul(product3, tf.transpose(inputs_col))
        outputs.append(self.act(rec))
        return outputs


class DistMultDecoder(MultiLayer):
    """DistMult Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(DistMultDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):

            tmp = inits.weight_variable_glorot(
                input_dim, 1, name='relation' )
            self.vars['relation' ] = tf.reshape(tmp, [-1])

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []

        inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
        inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
        relation = tf.diag(self.vars['relation'])
        intermediate_product = tf.matmul(inputs_row, relation)
        rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
        outputs.append(self.act(rec))
        return outputs


class BilinearDecoder(MultiLayer):
    """Bilinear Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(BilinearDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        with tf.variable_scope('%s_vars' % self.name):

            self.vars['relation' ] = inits.weight_variable_glorot(
                    input_dim, input_dim, name='relation' )

    def _call(self, inputs):
        i, j = self.edge_type
        outputs = []

        inputs_row = tf.nn.dropout(inputs[i], 1-self.dropout)
        inputs_col = tf.nn.dropout(inputs[j], 1-self.dropout)
        intermediate_product = tf.matmul(inputs_row, self.vars['relation'])
        rec = tf.matmul(intermediate_product, tf.transpose(inputs_col))
        outputs.append(self.act(rec))
        return outputs


class InnerProductDecoder(MultiLayer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def _call(self, inputs):
        outputs = []
        for k in range(1):
            inputs_row = tf.nn.dropout(inputs, 1-self.dropout)
            inputs_col = tf.nn.dropout(inputs, 1-self.dropout)
            rec = tf.matmul(inputs_row, tf.transpose(inputs_col))
            outputs.append(self.act(rec))
        return outputs




conv1d = tf.layers.conv1d


def attn_head(seq, out_sz, bias_mat, activation, in_drop=0.0, coef_drop=0.0, residual=False):

    with tf.name_scope('my_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)
        logits = f_1 + tf.transpose(f_2, [0, 2, 1])
        coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

        if coef_drop != 0.0:
            coefs = tf.nn.dropout(coefs, 1.0 - coef_drop)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        vals = tf.matmul(coefs, seq_fts)
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation


# Experimental sparse attention head (for running on datasets such as Pubmed)
# N.B. Because of limitations of current TF implementation, will work _only_ if batch_size = 1!
def sp_attn_head(seq, out_sz, adj_mat, activation, nb_nodes, in_drop=0.0, coef_drop=0.0, residual=False):
    with tf.name_scope('sp_attn'):
        if in_drop != 0.0:
            seq = tf.nn.dropout(seq, 1.0 - in_drop)

        seq_fts = tf.layers.conv1d(seq, out_sz, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = adj_mat * f_1
        f_2 = adj_mat * tf.transpose(f_2, [1, 0])

        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        if coef_drop != 0.0:
            coefs = tf.SparseTensor(indices=coefs.indices,
                                    values=tf.nn.dropout(coefs.values, 1.0 - coef_drop),
                                    dense_shape=coefs.dense_shape)
        if in_drop != 0.0:
            seq_fts = tf.nn.dropout(seq_fts, 1.0 - in_drop)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.squeeze(seq_fts)
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.expand_dims(vals, axis=0)
        vals.set_shape([1, nb_nodes, out_sz])
        ret = tf.contrib.layers.bias_add(vals)

        # residual connection
        if residual:
            if seq.shape[-1] != ret.shape[-1]:
                ret = ret + conv1d(seq, ret.shape[-1], 1)  # activation
            else:
                ret = ret + seq

        return activation(ret)  # activation

def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas