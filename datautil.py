import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
# from scipy.sparse import csr_matrix, csc_matrix
from scipy import sparse
from sklearn.metrics import roc_curve
from sklearn import metrics
import random
from numpy import zeros, ones
import numpy as np

def getData(path):
    node1 = []
    node2 = []
    edge = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n').split('\t')
            # print(line)
            node1.append(line[0])
            node2.append(line[1])
            edge.append((line[0], line[1]))
    return node1, node2, edge


def get_adj(path):
    adj = []
    with open(path, "r") as f:
        for line in f.readlines():
            l = line.strip("\n").split(' ')
            adj.append(l)

    adj = np.asarray(adj, dtype=float)
    #print(adj.shape)

    return adj


def get_metapath_adj():
    G = nx.Graph()
    nodes = []
    for i in range(0, 9031):
        nodes.append(str(i))

    G.add_nodes_from(nodes)
    (node1, node2, edge) = getData('./0416/data/dis_dis.tsv')
    G.add_edges_from(edge)
    (node1, node2, edge) = getData('./0416/data/dis_dru.tsv')
    G.add_edges_from(edge)
    (node1, node2, edge) = getData('./0416/data/dru_dru.tsv')
    G.add_edges_from(edge)
    A = np.array(nx.adjacency_matrix(G).todense())

    dis_dis_adj = np.zeros((6933, 6933))
    dru_dru_adj = np.zeros((2098, 2098))

    for i in range(0, 6933):
        for j in range(0, 6933):
            for k in range(6933, 9031):
                if (A[i][k] == 1 and A[j][k] == 1):
                    dis_dis_adj[i][j] = 1
                    break
        print(str(i))

    for i in range(6933, 9031):
        for j in range(6933, 9031):
            for k in range(0, 6933):
                if (A[i][k] == 1 and A[j][k] == 1):
                    dru_dru_adj[i - 6933][j - 6933] = 1
                    break
        print(str(i))
    dis_dru_adj = A[0:6933, 6933:]

    return dis_dis_adj, dis_dru_adj, dru_dru_adj


# a,b,c = get_metapath_adj()
# print('4')
def get_graph():
    G = nx.Graph()
    nodes = []
    for i in range(0, 9031):
        nodes.append(str(i))

    G.add_nodes_from(nodes)
    (node1, node2, edge) = getData('./0416/data/dis_dis.tsv')
    G.add_edges_from(edge)
    (node1, node2, edge) = getData('./0416/data/dis_dru.tsv')
    G.add_edges_from(edge)
    (node1, node2, edge) = getData('./0416/data/dru_dru.tsv')
    G.add_edges_from(edge)

    return G


def drawroc(y_test, y_score, name):
    fpr, tpr, thre = roc_curve(y_test, y_score)
    ##计算auc的值，就是roc曲线下的面积
    auc = metrics.auc(fpr, tpr)
    ##画图
    plt.plot(fpr, tpr, color='darkred', label='roc area:(%0.2f)' % auc)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.title('roc_curve')
    plt.legend(loc='lower right')
    plt.savefig('./result/auc'+ name + '.png')
    plt.show()
    plt.close()


def savelabelresult(labels_all, preds_all,embeding, name):
    np.savetxt("./result/" + name + "labels_all.txt", labels_all)
    np.savetxt("./result/" + name + "preds_all.txt", preds_all)
    np.savetxt("./result/" + name + "embeding.txt", embeding)


def adjtoedgelist(adj):
    adj = adj.A
    print(type(adj))
    edgelist = []
    for i in range(adj.shape[0]):
        for j in range(adj.shape[1]):
            if adj[i][j] != 0:
                edgelist.append([i, j])

    return edgelist


def graph_to_adj(edgelist):
    G = nx.Graph()
    nodes = []
    for i in range(0,6311):
        nodes.append(i)

    G.add_nodes_from(nodes)
    G.add_edges_from(edgelist)
    A = nx.adjacency_matrix(G)

    return A


def get_mateneigh_adj(mat):
    k = 1
    n = mat.shape[0]
    print(n)
    mateneigh_adj = np.zeros([n, n],dtype=float)
    c=0
    for i in range(n):
        for j in range(i+1,n):
            if np.dot(mat[i], mat[j]) > k:
                # print(np.dot(mat[i], mat[j]))
                mateneigh_adj[i][j] = 1.0
                mateneigh_adj[i][j] = 1.0
                c+=1
    print(c)
    return mateneigh_adj

def hanloaddata():

    pathus = "./dataset/mat_drug_disease.txt"
    pathup = "./dataset/mat_drug_protein.txt"
    pathsp = "./dataset/mat_protein_disease.txt"
    pathuu = "./dataset/mat_drug_drug.txt"

    dru_dis_adj = get_adj(pathus)
    dru_pro_adj = get_adj(pathup)
    dis_pro_adj = get_adj(pathsp)
    dru_dru_adj = get_adj(pathuu)
    dru_dis_mateneigh_adj = get_mateneigh_adj(dru_dis_adj)
    dis_dru_mateneigh_adj = get_mateneigh_adj(dru_dis_adj.T)
    dru_pro_mateneigh_adj = get_mateneigh_adj(dru_pro_adj)
    dis_pro_mateneigh_adj = get_mateneigh_adj(dis_pro_adj.T)
#     print('usu:',dru_dis_mateneigh_adj.sum())
# #     print('sus:',dis_dru_mateneigh_adj.sum())
# #     print('upu:',dru_pro_mateneigh_adj.sum())
# #     print('sps:',dis_pro_mateneigh_adj.sum())

    temp1 = np.concatenate((dru_dru_adj , dru_dis_adj.T), axis=0)
    temp2 = np.concatenate((dru_dis_adj, np.zeros((5603,5603),dtype=float)), axis=0)
    drudis = np.concatenate((temp1, temp2), axis=1)

    temp1 = np.concatenate((dru_pro_mateneigh_adj,dru_dis_adj.T),axis=0)
    temp2 = np.concatenate((dru_dis_adj, dis_pro_mateneigh_adj), axis=0)
    upusps = np.concatenate((temp1, temp2), axis=1)

    temp1 = np.concatenate((dru_dis_mateneigh_adj, dru_dis_adj.T), axis=0)
    temp2 = np.concatenate((dru_dis_adj, dis_dru_mateneigh_adj), axis=0)
    ususus = np.concatenate((temp1, temp2), axis=1)

    print(drudis.shape)
    print(upusps.shape)
    print(ususus.shape)
    drudis = sparse.csr_matrix(drudis)
    upusps = sparse.csr_matrix(upusps)
    ususus = sparse.csr_matrix(ususus)

    return drudis, upusps, ususus



def gatloaddata():
    pathus = "./dataset/mat_drug_disease.txt"
    dru_dis_adj = get_adj(pathus)
    temp1 = np.concatenate((np.zeros((708,708),dtype=float), dru_dis_adj.T), axis=0)
    temp2 = np.concatenate((dru_dis_adj, np.zeros((5603,5603),dtype=float)), axis=0)
    drudis = np.concatenate((temp1, temp2), axis=1)

    print(drudis.shape)

    drudis = sparse.csr_matrix(drudis)


    return drudis


def gcnloaddata():
    pathus = "./dataset/mat_drug_disease.txt"
    dru_dis_adj = get_adj(pathus)
    temp1 = np.concatenate((np.zeros((708,708),dtype=float), dru_dis_adj.T), axis=0)
    temp2 = np.concatenate((dru_dis_adj, np.zeros((5603,5603),dtype=float)), axis=0)
    drudis = np.concatenate((temp1, temp2), axis=1)

    print(drudis.shape)

    drudis = sparse.csr_matrix(drudis)
    return drudis











