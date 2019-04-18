import os
import numpy as np
from scipy.sparse import csr_matrix
import math
import random
import scipy.sparse
from sklearn import preprocessing, svm, metrics, tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neural_network import MLPClassifier



def readGraph(category, dataname, numv):
    '''
    read dealt graph
    :param category: the category of dataset
    :param dataname: the name of dataset
    :param v: the number of nodes
    :return: the matrix representation of dataset
    '''
    row = []
    col = []
    data = []
    filegraph = open('./' + category + '/' + dataname + '.txt')
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        row.append(line[0])
        col.append(line[1])
        data.append(1.0)
    graph = csr_matrix((data, (row, col)), shape=(numv, numv))
    filegraph.close()
    return graph, graph.sum(axis=0)


def CN(graph):
    '''
    calculate graph * graph
    :param graph: the matrix representation of graph
    :param flag: 1 means |V| < 30000, 0 means |V| >= 30000
    :return: graph * graph
    '''
    return graph.dot(graph)


def AA(graph, degree, category, dataname):
    row = []
    col = []
    data = []
    filegraph = open('./finaldata2/' + category + '/' + dataname + '.txt')
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        row.append(line[0])
        col.append(line[1])
        if degree[line[0]] == 1:
            data.append(0)
        else:
            data.append(1.0 / math.log(degree[line[0]]))
    log_index = csr_matrix((data, (row, col)), shape=(graph.shape[0], graph.shape[0]))
    filegraph.close()
    return graph.dot(log_index)


def RA(graph, degree, category, dataname):
    row = []
    col = []
    data = []
    filegraph = open('./finaldata2/' + category + '/' + dataname + '.txt')
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        row.append(line[0])
        col.append(line[1])
        data.append(1.0 / degree[line[0]])
    index = csr_matrix((data, (row, col)), shape=(graph.shape[0], graph.shape[0]))
    filegraph.close()
    return graph.dot(index)


def LNB(graph, cn, f, degree, category, dataname):
    alpha = graph.shape[0] * (graph.shape[0] - 1) * 0.5 / graph.getnnz() - 1
    tri = graph.dot(cn).diagonal() * 0.5
    tri_max = np.multiply(np.mat([degree]), (np.mat([degree]) - np.mat(np.ones((1, len(degree)))))) * 0.5
    R_w = np.array((np.mat(tri) + np.ones((1, len(tri)))) / (tri_max + np.ones((1, len(tri)))))[0]
    row = []
    col = []
    data = []
    filegraph = open('./finaldata2/' + category + '/' + dataname + '.txt')
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        row.append(line[0])
        col.append(line[1])
        if f == 'CN':
            data.append(math.log(alpha * R_w[line[0]]))
        if f == 'AA':
            if degree[line[0]] == 1:
                data.append(0)
            else:
                data.append(1.0 / math.log(degree[line[0]]) * math.log(alpha * R_w[line[0]]))
        if f == 'RA':
            data.append(1.0 / degree[line[0]] * math.log(alpha * R_w[line[0]]))
    filegraph.close()
    index = csr_matrix((data, (row, col)), shape=(graph.shape[0], graph.shape[0]))
    return graph.dot(index)


def LPI(graph, cn):
    return cn + 0.001 * cn.dot(graph)


def LRW(graph, degree, category, dataname, steps, lam):
    row = []
    col = []
    data = []
    filegraph = open('./finaldata2/' + category + '/' + dataname + '.txt')
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        row.append(line[0])
        col.append(line[1])
        data.append(1.0 / degree[line[0]])
    filegraph.close()
    index = csr_matrix((data, (row, col)), shape=(graph.shape[0], graph.shape[0]))
    I = csr_matrix(scipy.sparse.identity(graph.shape[0]))
    sims = I.copy()
    for i in range(0, steps):
        sims = (1 - lam) * I + lam * index.transpose().dot(sims)
    sims = sims + sims.transpose()
    return sims


def getIndex(x, y, cn, degree, aa, ra, lnb_cn, lnb_aa, lnb_ra, lpi, lrw):
    if cn[x, y] == 0:
        return [0, 0, 0, 0, 0, 0, 0, aa[x, y], ra[x, y], degree[x] * degree[y], lnb_cn[x, y], lnb_aa[x, y],
                lnb_ra[x, y], lpi[x, y], lrw[x, y]]
    else:
        salton = cn[x, y] / math.sqrt(degree[x] * degree[y])
        jaccard = cn[x, y] / math.sqrt(degree[x] + degree[y] - cn[x, y])
        sorensen = 2 * cn[x, y] / (degree[x] + degree[y])
        hpi = cn[x, y] / min(degree[x], degree[y])
        hdi = cn[x, y] / max(degree[x], degree[y])
        lhn1 = cn[x, y] / (degree[x] * degree[y])
        pa = degree[x] * degree[y]
        return [cn[x, y], salton, jaccard, sorensen, hpi, hdi, lhn1, aa[x, y], ra[x, y], pa, lnb_cn[x, y], lnb_aa[x, y],
                lnb_ra[x, y], lpi[x, y], lrw[x, y]]


def conductNegTrain(graph, category, dataname):
    filegraph = open('./finaldata2/' + category + '/' + dataname + '_neg.txt')
    test = {}
    neg_train = {}
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        if line[0] not in test:
            test[line[0]] = set()
        if line[1] not in test:
            test[line[1]] = set()
        test[line[0]].add(line[1])
        test[line[1]].add(line[0])
    filegraph.close()
    filegraph = open('./finaldata2/' + category + '/' + dataname + '_pos.txt')
    for line in filegraph:
        line = list(map(int, line.strip().split('\t')))
        if line[0] not in test:
            test[line[0]] = set()
        if line[1] not in test:
            test[line[1]] = set()
        test[line[0]].add(line[1])
        test[line[1]].add(line[0])
    filegraph.close()
    count = 0
    while count < graph.getnnz():
        x = random.randint(0, graph.get_shape()[0] - 1)
        y = random.randint(0, graph.get_shape()[0] - 1)
        if x in test:
            if y in test[x]:
                continue
        if y in test:
            if x in test[y]:
                continue
        if x not in neg_train:
            neg_train[x] = set()
        neg_train[x].add(y)
        count += 1
    return neg_train


def generateFeatures():
    # numV['CA-CondMat'] = 23133
    # numV['CA-GrQc'] = 5241
    # numV['CA-HepPh'] = 12006
    # numV['CA-HepTh'] = 9875
    # numV['CA-AstroPh'] = 18771
    # numV['caida'] = 26475
    # numV['gnutella'] = 62586
    # numV['route'] = 6474
    # numV['dnc'] = 906
    # numV['douban'] = 154908
    # numV['epinions'] = 75879
    # numV['facebook'] = 2888
    # numV['gowalla'] = 196591
    # numV['gplus'] = 23628
    # numV['hamster'] = 2426
    # numV['livemocha'] = 104103
    # numV['pretty'] = 10680
    # numV['advogato'] = 5155
    categories = ['humanreal', 'infrastructure', 'interaction', 'metabolic']
    for category in categories:
        for root, dirs, files in os.walk('./data2/' + category):
            for file in files:
                dataname = os.path.splitext(file)[0]
                print (dataname)
                graph, degree = readGraph(category, dataname, numV[dataname])
                degree = list(map(int, degree.getA()[0]))
                cn = CN(graph)
                aa = AA(graph, degree, category, dataname)
                ra = RA(graph, degree, category, dataname)
                lnb_cn = LNB(graph, cn, 'CN', degree, category, dataname)
                lnb_aa = LNB(graph, cn, 'AA', degree, category, dataname)
                lnb_ra = LNB(graph, cn, 'RA', degree, category, dataname)
                lpi = LPI(graph, cn)
                lrw = LRW(graph, degree, category, dataname, 5, 0.85)

                neg_train = conductNegTrain(graph, category, dataname)
                X_train = []
                Y_train = []
                for key in neg_train:
                    for value in neg_train[key]:
                        X_train.append(getIndex(key, value, cn, degree, aa, ra, lnb_cn, lnb_aa, lnb_ra, lpi, lrw))
                        Y_train.append(0)
                tuple = graph.nonzero()
                pos_train = {}
                for i in range(0, len(tuple[0])):
                    if tuple[0][i] in pos_train and tuple[1][i] in pos_train[tuple[0][i]]:
                        continue
                    if tuple[1][i] in pos_train and tuple[0][i] in pos_train[tuple[1][i]]:
                        continue
                    X_train.append(getIndex(tuple[0][i], tuple[1][i], cn, degree, aa, ra, lnb_cn, lnb_aa, lnb_ra, lpi, lrw))
                    Y_train.append(1)
                    if tuple[0][i] not in pos_train:
                        pos_train[tuple[0][i]] = set()
                    if tuple[1][i] not in pos_train:
                        pos_train[tuple[1][i]] = set()
                    pos_train[tuple[0][i]].add(tuple[1][i])
                    pos_train[tuple[1][i]].add(tuple[0][i])

                X_test = []
                Y_test = []
                repeatSamples = 0
                filegraph = open('./finaldata2/' + category + '/' + dataname + '_neg.txt')
                for line in filegraph:
                    line = list(map(int, line.strip().split('\t')))
                    X_test.append(getIndex(line[0], line[1], cn, degree, aa, ra, lnb_cn, lnb_aa, lnb_ra, lpi, lrw))
                    Y_test.append(0)
                filegraph.close()
                filegraph = open('./finaldata2/' + category + '/' + dataname + '_pos.txt')
                for line in filegraph:
                    line = list(map(int, line.strip().split('\t')))
                    X_test.append(getIndex(line[0], line[1], cn, degree, aa, ra, lnb_cn, lnb_aa, lnb_ra, lpi, lrw))
                    Y_test.append(1)
                filegraph.close()
                print repeatSamples
                scaler = preprocessing.StandardScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                print ('features got')
                SVM(X_train, Y_train, X_test, Y_test)
                KNN(X_train, Y_train, X_test, Y_test)
                DT(X_train, Y_train, X_test, Y_test)
                Bayes(X_train, Y_train, X_test, Y_test)
                LR(X_train, Y_train, X_test, Y_test)
                MLP(X_train, Y_train, X_test, Y_test)


def SVM(X_train, Y_train, X_test, Y_test):
        bestsvm = svm.SVC(kernel='rbf', probability=True)
        print ('default')
        bestsvm.fit(X_train, Y_train)
        proba = bestsvm.predict_proba(X_test)
        result = [x[1] for x in proba]
        print ('svm -auc:', metrics.roc_auc_score(Y_test, result))


def KNN(X_train, Y_train, X_test, Y_test):
    bestn = 15
    bestknn = KNeighborsClassifier(bestn)
    bestknn.fit(X_train, Y_train)
    proba = bestknn.predict_proba(X_test)
    result = [x[1] for x in proba]
    print ('knn -auc:', metrics.roc_auc_score(Y_test, result))


def DT(X_train, Y_train, X_test, Y_test):
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, Y_train)
    proba = clf.predict_proba(X_test)
    result = [x[1] for x in proba]
    print ('decision tree -auc:', metrics.roc_auc_score(Y_test, result))


def Bayes(X_train, Y_train, X_test, Y_test):
    clf = GaussianNB(priors=[0.5, 0.5])
    clf.fit(X_train, Y_train)
    proba = clf.predict_proba(X_test)
    result = [x[1] for x in proba]
    print ('bayes -auc:', metrics.roc_auc_score(Y_test, result))


def LR(X_train, Y_train, X_test, Y_test):
    lr = LogisticRegressionCV(cv=10, scoring='roc_auc', penalty='l2', solver='lbfgs')
    lr.fit(X_train, Y_train)
    proba = lr.predict_proba(X_test)
    result = [x[1] for x in proba]
    print ('logistic regression -auc:', metrics.roc_auc_score(Y_test, result))


def MLP(X_train, Y_train, X_test, Y_test):
    clf = MLPClassifier(solver='adam', activation='logistic')
    clf.fit(X_train, Y_train)
    proba = clf.predict_proba(X_test)
    result = [x[1] for x in proba]
    print ('mlp -auc:', metrics.roc_auc_score(Y_test, result))

