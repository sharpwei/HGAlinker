from collections import defaultdict

import tensorflow as tf
import numpy as np
from .layers import GraphConvolutionMulti, GraphConvolutionSparseMulti, \
    DistMultDecoder, InnerProductDecoder, DEDICOMDecoder, BilinearDecoder
from .gatlink import GAT
from .hanlink import HeteGAT_multi


flags = tf.app.flags
FLAGS = flags.FLAGS
residual = False
nonlinearity = tf.nn.elu
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer



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


class HanModel(Model):
    def __init__(self, placeholders, num_feat, nonzero_feat, matenum,decoders, hid_units =[8], mp_att_size= 128,  **kwargs):
        super(HanModel, self).__init__(**kwargs)
        # self.edge_types = edge_types
        self.matenum = matenum
        self.num_edge_types = 1
        self.num_obj_types = 1
        self.decoders = decoders
        self.inputs = [placeholders['feat_%d' %i] for i in range(self.matenum)]
        self.input_dim = num_feat
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.adj_mats =[placeholders['adj_mats_%d' %i] for i in range(self.matenum)]
        self.attn_drop = placeholders['attn_drop']
        self.ffd_drop = placeholders['ffd_drop']
        self.hid_units = hid_units
        self.mp_att_size =mp_att_size
        self.build()


    def _build(self):
        self.embeddings_reltyp = defaultdict(list)

        logits, self.final_embedding, self.att_val = HeteGAT_multi.inference(self.inputs, FLAGS.hidden2, nb_nodes=10000, training = True,
                                                           attn_drop = self.attn_drop, ffd_drop=self.ffd_drop,bias_mat_list=self.adj_mats,
                                                           hid_units=self.hid_units, n_heads=n_heads,
                                                           residual=residual, activation=nonlinearity,)

        self.embeddings_reltyp[0].append(logits)


        self.embeddings = [None] * self.num_obj_types

        for i, embeds in self.embeddings_reltyp.items():
            # self.embeddings[i] = tf.nn.relu(tf.add_n(embeds))
            self.embeddings[i] = tf.add_n(embeds)
        # print(self.embeddings )
        self.embeddings = tf.squeeze(self.embeddings)
        # print(self.embeddings)
        self.edge_type2decoder ={}
        decoder = self.decoders


        if decoder == 'innerproduct':
            self.edge_type2decoder[1]= InnerProductDecoder(
                input_dim=FLAGS.hidden2, logging=self.logging,
                act=lambda x: x, dropout=self.dropout)


        self.latent_inters = []
        self.latent_varies = []


        decoder = self.decoders
        for k in range(1):
            if decoder == 'innerproduct':
                glb = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'distmult':
                glb = tf.diag(self.edge_type2decoder[2].vars['relation' ])
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'bilinear':
                glb = self.edge_type2decoder[3].vars['relation']
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'dedicom':
                glb = self.edge_type2decoder[4].vars['global_interaction']
                loc = tf.diag(self.edge_type2decoder.vars['local_variation'])
            else:
               raise ValueError('Unknown decoder type')

        self.latent_inters.append(glb)
        self.latent_varies.append(loc)




class GATModel(Model):
    def __init__(self, placeholders, num_feat, nonzero_feat, decoders, **kwargs):
        super(GATModel, self).__init__(**kwargs)
        # self.edge_types = edge_types
        self.num_edge_types = 1
        self.num_obj_types = 1
        self.decoders = decoders
        self.inputs = placeholders['feat']
        self.input_dim = num_feat
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.adj_mats =placeholders['adj_mats' ]
        self.attn_drop = placeholders['attn_drop']
        self.ffd_drop = placeholders['ffd_drop']
        self.build()


    def _build(self):
        print("222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222222")
        self.embeddings_reltyp = defaultdict(list)

        print(self.inputs)
        print("self.inputs[j]")
        self.embeddings_reltyp[0].append(GAT.inference(self.inputs, FLAGS.hidden2, nb_nodes=10000, training = True,
                                     attn_drop = self.attn_drop, ffd_drop=self.ffd_drop,bias_mat=self.adj_mats,
                                     hid_units=hid_units, n_heads=n_heads,
                                     residual=residual, activation=nonlinearity))

        self.embeddings = [None] * self.num_obj_types

        for i, embeds in self.embeddings_reltyp.items():
            # self.embeddings[i] = tf.nn.relu(tf.add_n(embeds))
            self.embeddings[i] = tf.add_n(embeds)
        # print(self.embeddings )
        self.embeddings = tf.squeeze(self.embeddings)
        # print(self.embeddings)
        self.edge_type2decoder ={}
        decoder = self.decoders


        if decoder == 'innerproduct':
            self.edge_type2decoder[1]= InnerProductDecoder(
                input_dim=FLAGS.hidden2, logging=self.logging,
                act=lambda x: x, dropout=self.dropout)


        self.latent_inters = []
        self.latent_varies = []


        decoder = self.decoders
        for k in range(1):
            if decoder == 'innerproduct':
                glb = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'distmult':
                glb = tf.diag(self.edge_type2decoder[2].vars['relation' ])
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'bilinear':
                glb = self.edge_type2decoder[3].vars['relation']
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'dedicom':
                glb = self.edge_type2decoder[4].vars['global_interaction']
                loc = tf.diag(self.edge_type2decoder.vars['local_variation'])
            else:
               raise ValueError('Unknown decoder type')

        self.latent_inters.append(glb)
        self.latent_varies.append(loc)


class GCNModel(Model):
    def __init__(self, placeholders, num_feat, nonzero_feat, decoders, **kwargs):
        super(GCNModel, self).__init__(**kwargs)

        self.num_edge_types = 1
        self.num_obj_types = 1
        self.decoders = decoders
        self.inputs = placeholders['feat']
        self.input_dim = num_feat
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.adj_mats =placeholders['adj_mats' ]
        self.attn_drop = placeholders['attn_drop']
        self.ffd_drop = placeholders['ffd_drop']
        self.build()

    def _build(self):
        self.hidden1 = defaultdict(list)
        print(type(self.inputs))
        # for i, j in self.edge_types:
        self.hidden1[0].append(GraphConvolutionSparseMulti(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1,
                adj_mats=self.adj_mats, nonzero_feat=self.nonzero_feat,
                act=lambda x: x, dropout=self.dropout,
                logging=self.logging)(self.inputs))

        for i, hid1 in self.hidden1.items():
            self.hidden1[i] = tf.nn.relu(tf.add_n(hid1))

        self.embeddings_reltyp = defaultdict(list)

        self.embeddings_reltyp[0].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden1, output_dim=FLAGS.hidden2,
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.hidden1[0]))

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp.items():
            # self.embeddings[i] = tf.nn.relu(tf.add_n(embeds))
            self.embeddings[i] = tf.add_n(embeds)

        self.latent_inters = []
        self.latent_varies = []

        decoder = self.decoders
        for k in range(1):
            if decoder == 'innerproduct':
                glb = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'distmult':
                glb = tf.diag(self.edge_type2decoder[2].vars['relation'])
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'bilinear':
                glb = self.edge_type2decoder[3].vars['relation']
                loc = tf.eye(FLAGS.hidden2, FLAGS.hidden2)
            elif decoder == 'dedicom':
                glb = self.edge_type2decoder[4].vars['global_interaction']
                loc = tf.diag(self.edge_type2decoder.vars['local_variation'])
            else:
                raise ValueError('Unknown decoder type')

        self.latent_inters.append(glb)
        self.latent_varies.append(loc)


