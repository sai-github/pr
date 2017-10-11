#helper functions

def ck_gamma(gamma_old, gamma_new, K):
    print("Old : {}, New : {}".format(sum(gamma_old[:,K-1]), sum(gamma_new[:,K-1])))