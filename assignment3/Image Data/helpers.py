import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

import itertools

from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal


#helper functions

# generalised functions
def multivariate_gaussian(pos, mu, Sigma):
    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    return np.exp(-fac / 2) / N
    
def cal_pi_k(X, clusters):
    '''Calculate pi_k from initial KMeans step '''
    pi_k = []
    for i in clusters:
        pi_k.append(clusters[i].shape[0])
    return np.array(pi_k)/X.shape[0]

def recal_pi_k(gamma):
    '''Calculate pi_k(new) after an iteration from Expectation step'''
    K = gamma.shape[1]
    n_size = gamma.shape[0]
    mix = [0]*K
    for i in range(K):
        mix[i] = sum(gamma[:,i])/n_size
        
    return np.array(mix)

def cal_mu(X, gamma):
    '''Calculate Mu(new) after an iteration from Expectation step or initial KMeans step '''
    
    K = gamma.shape[1]
    
    new_centroids = np.empty((K,X.shape[1]))
    
    for k in range(K):
        # Denominator
        temp_sum= sum(gamma[:,k])
        temp_s = np.array([0]*X.shape[1], dtype='float64')
        
        # Numerator
        for n in range(X.shape[0]):
            #print(gamma.shape, X[n].shape)
            temp_s += gamma[n][k]*X[n]
        if temp_sum > 1:
            new_centroids[k] = temp_s/temp_sum
    
    return new_centroids

def cal_sigma(X, gamma, mu):
    
    K = gamma.shape[1]
    feature_size = X.shape[1]
    new_sigma = np.empty((K , feature_size, feature_size))
    
    for k in range(K):
        # Denominator
        temp_sum= sum(gamma[:,k])
        temp_s = np.array([[0]*feature_size]*feature_size, dtype='float64')
        # Numerator
        for n in range(X.shape[0]):
            t = (X[n] - mu[k]).reshape(1,feature_size)
            temp_s += gamma[n][k] * np.matmul(np.transpose(t), t)

        new_sigma[k] = temp_s / temp_sum
        
    return new_sigma

def build_gamma(X, mix, mu, sigma):
    
    K = mix.shape[0]
    
    tmp = np.zeros([X.shape[0], K])
    
    for i in range(K):
        mean = mu[i]
        cov = sigma[i]        
        tmp[:, i] = mix[i]*multivariate_normal.pdf(X, mean, cov) # using inbuilt pdf
        #tmp[:, i] = mix[i]*multivariate_gaussian(U, mean, cov)  # self made pdf function
        
    gamma = np.zeros([X.shape[0], K])

    # Instead of using two data structures temp and gamma we can use only 1
    
    for i in range(X.shape[0]):
        temp_sum = sum(tmp[i, :])
        for j in range(K):
            gamma[i][j] = tmp[i][j] / temp_sum
            
    return gamma

def threshold(value, limit = 10**-3):
    return (abs(value)<=limit)

def em_repeat(gmm_initial, X):
    old_pi = gmm_initial.getpi()
    old_mu = gmm_initial.getmu()
    old_sigma = gmm_initial.getsigma()
    
    gamma = build_gamma(X, old_pi, old_mu, old_sigma)
    
    new_pi = recal_pi_k(gamma)
    new_mu = cal_mu(X, gamma)
    new_sigma = cal_sigma(X, gamma, new_mu)
    
    new_gmm = gmm(new_mu, new_sigma, new_pi)
    
    return new_gmm

def em_repeat_times(gmm, X, times=2):
    for _ in range(times):
        gmm = em_repeat(gmm, X)
        
    return gmm

def normalize(X):
    return (X-min(X))/(max(X)-min(X))

def classify(gA, gB, X):
    return 1 if (gA.predict_scores(X)>gB.predict_scores(X)).all() else 2

#Classes
# GMM class

class gmm:
    def __init__(self, list_mu, list_sigma, list_pi):
        self.list_mu = list_mu
        self.list_sigma = list_sigma
        self.list_pi = list_pi
        
    def predict_scores(self, X):
        feature_size = self.list_mu.shape[1]
        X = X.reshape(-1,feature_size)
        scores = np.empty((X.shape[0], 1), dtype = "float64")
        
        for n in range(X.shape[0]):
            temp_s = 0
            for k in range(self.list_mu.shape[0]):
                rv = multivariate_normal(self.list_mu[k], self.list_sigma[k])
                #print(X[n], X[n].shape)
                temp_s += self.list_pi[k] * rv.pdf(X[n])
            scores[n] = temp_s
        return scores
    
    def getpi(self):
        return self.list_pi
    
    def getmu(self):
        return self.list_mu
    
    def getsigma(self):
        return self.list_sigma
    

#Plots functions

def plot_elbow(X, K=10):
    
    distortions = []
    for k in range(1,K):
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(range(1,K), distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()
    
    cluster_size = K
    clf = KMeans(n_clusters = cluster_size)
    clf.fit(X)

    # Gives the final cetnroids of each cluster
    centroids = clf.cluster_centers_

    # Label of each data-point
    labels = clf.labels_

    
    clusters = {}
    #initialize
    for i in range(cluster_size):
        clusters[i] = []

    for i,j in zip(X, labels):
        clusters[j].append(i)

    for i in clusters:
        clusters[i] = np.array(clusters[i])
        


#pipeline
def pipeline(X, K=10, error = 0.1):
    clf = KMeans(n_clusters = K)
    clf.fit(X)

    # Gives the final cetnroids of each cluster
    centroids = clf.cluster_centers_
    labels = clf.labels_
    
    clusters = {}
    #initialize
    for i in range(K):
        clusters[i] = []

    for i,j in zip(X, labels):
        clusters[j].append(i)

    for i in clusters:
        clusters[i] = np.array(clusters[i])
    
    
    #Start of EM Step
    initial_sigma = []
    for index in range(K):
        initial_sigma.append(np.cov(np.transpose(clusters[index])))

   # Initialization
    feature_size = initial_sigma[0].shape[0]
    initial_sigma = np.array(initial_sigma).reshape(-1, feature_size, feature_size)
    initial_mu = centroids
    initial_pi = cal_pi_k(X, clusters)
    

    g1 = gmm(initial_mu, initial_sigma, initial_pi)
    s1 = g1.predict_scores(X)
    
    g2 = em_repeat_times(g1, X, times = 1)
    s2 = g2.predict_scores(X)

    count_itr = 0
    while abs(sum(np.log(s2)) - sum(np.log(s1)))> error :
        s1 = g1.predict_scores(X)
        g2 = em_repeat_times(g1, X, times = 1)
        s2 = g2.predict_scores(X)
        g1 = g2
        count_itr+=1
        
    print("Iterations : {}".format(count_itr))
    return g2