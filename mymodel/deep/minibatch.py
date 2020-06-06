from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp

from ..utility import preprocessing

np.random.seed(123)


class HANEdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    assoc -- numpy array with target edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    """
    def __init__(self, adj_mats, feat, batch_size=100, val_test_size=0.01):
        self.adj_mats = adj_mats
        print("self.adj_mats")
        # print(self.adj_mats.todense())
        self.feat = feat
        self.batch_size = batch_size
        self.val_test_size = val_test_size
        self.iter = 0

        self.batch_num = 0
        self.current_edge_type_idx = 0

        self.train_edges = [None]
        self.val_edges = [None]
        self.test_edges = [None]
        self.test_edges_false = [None]
        self.val_edges_false = [None]

        # Function to build test and val sets with val_test_size positive links
        self.adj_train =[None]
        print("Minibatch edge ")
        self.mask_test_edges()

        print("Train edges=", "%04d" % len(self.train_edges))
        print("Val edges=", "%04d" % len(self.val_edges))
        print("Test edges=", "%04d" % len(self.test_edges))


    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        return preprocessing.sparse_to_tuple(adj_normalized)

    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)

    def mask_test_edges(self):
        edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats[0])
        num_test = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))
        num_val = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))

        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        val_edges = edges_all[val_edge_idx]

        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges_all[test_edge_idx]

        train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        test_edges_false = []

        while len(test_edges_false) < len(test_edges):
            if len(test_edges_false) % 1000 == 0:
                print("Constructing test edges=", "%04d/%04d" % (len(test_edges_false), len(test_edges)))

            idx_i = np.random.randint(0, self.adj_mats[0].shape[0])
            idx_j = np.random.randint(0, self.adj_mats[0].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if self._ismember([idx_i, idx_j], test_edges_false):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            if len(val_edges_false) % 1000 == 0:
                print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
            idx_i = np.random.randint(0, self.adj_mats[0].shape[0])
            idx_j = np.random.randint(0, self.adj_mats[0].shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if val_edges_false:
                if self._ismember([idx_i, idx_j], val_edges_false):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats[0].shape)
        self.adj_train = self.preprocess_graph(adj_train)
        print("adj_train")
        print(type(adj_train))
        self.train_edges = train_edges
        self.val_edges = val_edges
        self.val_edges_false = np.array(val_edges_false)
        self.test_edges = test_edges
        self.test_edges_false = np.array(test_edges_false)

    def end(self):
        # finished = len(self.freebatch_edge_types) == 0
        finished = self.batch_num * self.batch_size > len(self.train_edges) - self.batch_size + 1
        return finished

    def update_feed_dict(self, feed_dict, dropout, placeholders):
        # construct feed dictionary
        # feed_dict.update({placeholders['adj_mats']: self.adj_train})
        for i in range(len(self.adj_mats)):
            feed_dict.update({placeholders['adj_mats_%d' %i]: self.adj_mats[i].todense()[np.newaxis]})
        # print(self.adj_train.dtype)
        for i in range(len(self.feat)):
            feed_dict.update({placeholders['feat_%d' %i]: self.feat[i][np.newaxis]})

        feed_dict.update({placeholders['dropout']: dropout})
        feed_dict.update({placeholders['attn_drop']: 0.60})
        feed_dict.update({placeholders['ffd_drop']: 0.60})
        # feed_dict.update({placeholders['is_train']: True})

        return feed_dict

    def batch_feed_dict(self, batch_edges, placeholders):
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges})
        # print('55556565656')
        # print(batch_edges.shape)
        # feed_dict.update({placeholders['batch_edge_type_idx']: batch_edge_type})
        # feed_dict.update({placeholders['batch_row_edge_type']: self.idx2edge_type[batch_edge_type][0]})
        # feed_dict.update({placeholders['batch_col_edge_type']: self.idx2edge_type[batch_edge_type][1]})
        return feed_dict

    def next_minibatch_feed_dict(self, placeholders):
        """Select a random edge type and a batch of edges of the same type"""

        self.iter += 1
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        batch_edges = self.train_edges[start: start + self.batch_size]
        return self.batch_feed_dict(batch_edges, placeholders)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, placeholders, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, placeholders)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """

        self.iter = 0
        self.train_edges = np.random.permutation(self.train_edges)
        self.batch_num = 0


class GATEdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    assoc -- numpy array with target edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    """
    def __init__(self, adj_mats, feat, batch_size=100, val_test_size=0.01):
        self.adj_mats = adj_mats
        print("self.adj_mats")
        print(self.adj_mats.todense())
        self.feat = feat
        self.batch_size = batch_size
        self.val_test_size = val_test_size


        self.iter = 0

        self.batch_num = 0
        self.current_edge_type_idx = 0

        self.train_edges = [None]
        self.val_edges = [None]
        self.test_edges = [None]
        self.test_edges_false = [None]
        self.val_edges_false = [None]

        # Function to build test and val sets with val_test_size positive links
        self.adj_train =[None]
        print("Minibatch edge ")
        self.mask_test_edges()

        print("Train edges=", "%04d" % len(self.train_edges))
        print("Val edges=", "%04d" % len(self.val_edges))
        print("Test edges=", "%04d" % len(self.test_edges))


    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        return preprocessing.sparse_to_tuple(adj_normalized)

    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)

    def mask_test_edges(self):
        edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats)
        num_test = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))
        num_val = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))

        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        val_edges = edges_all[val_edge_idx]

        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges_all[test_edge_idx]

        train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        test_edges_false = []

        while len(test_edges_false) < len(test_edges):
            if len(test_edges_false) % 1000 == 0:
                print("Constructing test edges=", "%04d/%04d" % (len(test_edges_false), len(test_edges)))

            idx_i = np.random.randint(0, self.adj_mats.shape[0])
            idx_j = np.random.randint(0, self.adj_mats.shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if self._ismember([idx_i, idx_j], test_edges_false):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            if len(val_edges_false) % 1000 == 0:
                print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
            idx_i = np.random.randint(0, self.adj_mats.shape[0])
            idx_j = np.random.randint(0, self.adj_mats.shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if val_edges_false:
                if self._ismember([idx_i, idx_j], val_edges_false):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats.shape)
        self.adj_train = self.preprocess_graph(adj_train)
        print("adj_train")
        print(type( self.adj_train))
        self.train_edges = train_edges
        self.val_edges = val_edges
        self.val_edges_false = np.array(val_edges_false)
        self.test_edges = test_edges
        self.test_edges_false = np.array(test_edges_false)

    def end(self):
        # finished = len(self.freebatch_edge_types) == 0
        finished = self.batch_num * self.batch_size > len(self.train_edges) - self.batch_size + 1
        return finished

    def update_feed_dict(self, feed_dict, dropout, placeholders):

        feed_dict.update({placeholders['adj_mats']: self.adj_mats.todense()[np.newaxis]})
        feed_dict.update({placeholders['feat']: self.feat[np.newaxis]})
        feed_dict.update({placeholders['dropout']: dropout})
        feed_dict.update({placeholders['attn_drop']: 0.60})
        feed_dict.update({placeholders['ffd_drop']: 0.60})


        return feed_dict

    def batch_feed_dict(self, batch_edges, placeholders):
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges})
        return feed_dict

    def next_minibatch_feed_dict(self, placeholders):

        self.iter += 1
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        batch_edges = self.train_edges[start: start + self.batch_size]
        return self.batch_feed_dict(batch_edges, placeholders)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, placeholders, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, placeholders)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """

        self.iter = 0
        self.train_edges = np.random.permutation(self.train_edges)
        self.batch_num = 0



class GCNEdgeMinibatchIterator(object):
    """ This minibatch iterator iterates over batches of sampled edges or
    random pairs of co-occuring edges.
    assoc -- numpy array with target edges
    placeholders -- tensorflow placeholders object
    batch_size -- size of the minibatches
    """
    def __init__(self, adj_mats, feat, batch_size=100, val_test_size=0.01):
        self.adj_mats = adj_mats
        print("self.adj_mats")
        print(self.adj_mats.todense())
        self.feat = feat
        self.batch_size = batch_size
        self.val_test_size = val_test_size


        self.iter = 0

        self.batch_num = 0
        self.current_edge_type_idx = 0

        self.train_edges = [None]
        self.val_edges = [None]
        self.test_edges = [None]
        self.test_edges_false = [None]
        self.val_edges_false = [None]

        # Function to build test and val sets with val_test_size positive links
        self.adj_train =[None]
        print("Minibatch edge ")
        self.mask_test_edges()

        print("Train edges=", "%04d" % len(self.train_edges))
        print("Val edges=", "%04d" % len(self.val_edges))
        print("Test edges=", "%04d" % len(self.test_edges))


    def preprocess_graph(self, adj):
        adj = sp.coo_matrix(adj)
        if adj.shape[0] == adj.shape[1]:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            rowsum = np.array(adj.sum(1))
            colsum = np.array(adj.sum(0))
            rowdegree_mat_inv = sp.diags(np.nan_to_num(np.power(rowsum, -0.5)).flatten())
            coldegree_mat_inv = sp.diags(np.nan_to_num(np.power(colsum, -0.5)).flatten())
            adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        return preprocessing.sparse_to_tuple(adj_normalized)

    def _ismember(self, a, b):
        a = np.array(a)
        b = np.array(b)
        rows_close = np.all(a - b == 0, axis=1)
        return np.any(rows_close)

    def mask_test_edges(self):
        edges_all, _, _ = preprocessing.sparse_to_tuple(self.adj_mats)
        num_test = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))
        num_val = max(50, int(np.floor(edges_all.shape[0] * self.val_test_size)))

        all_edge_idx = list(range(edges_all.shape[0]))
        np.random.shuffle(all_edge_idx)

        val_edge_idx = all_edge_idx[:num_val]
        val_edges = edges_all[val_edge_idx]

        test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
        test_edges = edges_all[test_edge_idx]

        train_edges = np.delete(edges_all, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
        test_edges_false = []

        while len(test_edges_false) < len(test_edges):
            if len(test_edges_false) % 1000 == 0:
                print("Constructing test edges=", "%04d/%04d" % (len(test_edges_false), len(test_edges)))

            idx_i = np.random.randint(0, self.adj_mats.shape[0])
            idx_j = np.random.randint(0, self.adj_mats.shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if test_edges_false:
                if self._ismember([idx_i, idx_j], test_edges_false):
                    continue
            test_edges_false.append([idx_i, idx_j])

        val_edges_false = []
        while len(val_edges_false) < len(val_edges):
            if len(val_edges_false) % 1000 == 0:
                print("Constructing val edges=", "%04d/%04d" % (len(val_edges_false), len(val_edges)))
            idx_i = np.random.randint(0, self.adj_mats.shape[0])
            idx_j = np.random.randint(0, self.adj_mats.shape[1])
            if self._ismember([idx_i, idx_j], edges_all):
                continue
            if val_edges_false:
                if self._ismember([idx_i, idx_j], val_edges_false):
                    continue
            val_edges_false.append([idx_i, idx_j])

        # Re-build adj matrices
        data = np.ones(train_edges.shape[0])
        adj_train = sp.csr_matrix(
            (data, (train_edges[:, 0], train_edges[:, 1])),
            shape=self.adj_mats.shape)
        self.adj_train = self.preprocess_graph(adj_train)
        print("adj_train")
        print(type( self.adj_train))
        self.train_edges = train_edges
        self.val_edges = val_edges
        self.val_edges_false = np.array(val_edges_false)
        self.test_edges = test_edges
        self.test_edges_false = np.array(test_edges_false)

    def end(self):
        # finished = len(self.freebatch_edge_types) == 0
        finished = self.batch_num * self.batch_size > len(self.train_edges) - self.batch_size + 1
        return finished

    def update_feed_dict(self, feed_dict, dropout, placeholders):

        feed_dict.update({placeholders['adj_mats']: self.adj_train})
        feed_dict.update({placeholders['feat']: self.feat})
        feed_dict.update({placeholders['dropout']: dropout})
        feed_dict.update({placeholders['attn_drop']: 0.60})
        feed_dict.update({placeholders['ffd_drop']: 0.60})


        return feed_dict

    def batch_feed_dict(self, batch_edges, placeholders):
        feed_dict = dict()
        feed_dict.update({placeholders['batch']: batch_edges})
        return feed_dict

    def next_minibatch_feed_dict(self, placeholders):

        self.iter += 1
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        batch_edges = self.train_edges[start: start + self.batch_size]
        return self.batch_feed_dict(batch_edges, placeholders)

    def num_training_batches(self):
        return len(self.train_edges) // self.batch_size + 1

    def val_feed_dict(self, placeholders, size=None):
        edge_list = self.val_edges
        if size is None:
            return self.batch_feed_dict(edge_list, placeholders)
        else:
            ind = np.random.permutation(len(edge_list))
            val_edges = [edge_list[i] for i in ind[:min(size, len(ind))]]
            return self.batch_feed_dict(val_edges, placeholders)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """

        self.iter = 0
        self.train_edges = np.random.permutation(self.train_edges)
        self.batch_num = 0