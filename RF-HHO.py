import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestRegressor
from bunch import Bunch


def evaluate_model(model, x, y):
    pred = model.predict(x)
    r2 = metrics.r2_score(y, pred)
    corr_coef = np.corrcoef(y, pred)[0, 1]
    return r2,corr_coef


def load_npp():

    data_csv = pd.read_csv(r'MHW.csv')
    npp = Bunch()
    npp.data = nppdata(data_csv)
    npp.target = npptarget(data_csv)
    npp.DESCR = nppdescr(data_csv)
    npp.feature_names = feature_names()
    npp.target_names = target_names()

    return npp

def nppdata(data):

    data_r = data.iloc[:, 0:13]
    data_np = np.array(data_r)
    return data_np


def npptarget(data):

    data_b = data.iloc[:, -1]
    data_np = np.array(data_b)
    return data_np


def nppdescr(data):

    text = "The number of samples: {}:" \
           "The number of features: {}; The number of target: {}; No null data" \
           "".format(data.index.size, data.columns.size - 2, 1)
    return text


def feature_names():

    fnames=np.array(['Temperature', 'Fe', 'O2', 'Silicate', 'Nitrate', 'pH', 'Phosphate', 'Salinity', 'SPCO2', 'Chla', 'PAR', 'DOC', 'NH4', 'NPP'])
    return fnames


def target_names():

    tnames = ["NPP"]
    return tnames


def init_position(lb, ub, N, dim):

    X = np.zeros([N, dim], dtype='float')
    for i in range(N):
        for d in range(dim):
            X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()

    return X



def binary_conversion(X, thres, N, dim):

    Xbin = np.zeros([N, dim], dtype='int')
    for i in range(N):
        for d in range(dim):
            if X[i, d] > thres:
                Xbin[i, d] = 1
            else:
                Xbin[i, d] = 0

    return Xbin



def boundary(x, lb, ub):

    if x < lb:
        x = lb
    if x > ub:
        x = ub

    return x


def levy_distribution(beta, dim):

    nume = math.gamma(1 + beta) * np.sin(np.pi * beta / 2)
    deno = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
    sigma = (nume / deno) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    step = u / abs(v) ** (1 / beta)
    LF = 0.01 * step

    return LF


def error_rate(X_train, y_train, X_test, y_test, x, opts):
    if abs(x[0]) > 0:
        max_depth = int(abs(x[0])) + 5
    else:
        max_depth = int(abs(x[0])) + 8

    if abs(x[1]) > 0:
        n_estimators = int(abs(x[1]))*100
    else:
        n_estimators = int(abs(x[1]))+200

    rfr_model = RandomForestRegressor(max_depth=max_depth,
                                      n_estimators=n_estimators).fit(X_train, y_train)
    cv_accuracies = cross_val_score(rfr_model, X_test, y_test, cv=10,
                                    scoring='r2')

    accuracies = cv_accuracies.mean()
    fitness_value = 1 - accuracies

    return fitness_value


def Fun(X_train, y_train, X_test, y_test, x, opts):

    alpha = 0.99
    beta = 1 - alpha
    max_feat = len(x)
    num_feat = np.sum(x == 1)

    if num_feat == 0:
        cost = 1
    else:
        error = error_rate(X_train, y_train, X_test, y_test, x, opts)
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost


def jfs(X_train, y_train, X_test, y_test, opts):

    ub = 1
    lb = 0  #
    thres = 0.5
    beta = 1.5

    N = opts['N']
    max_iter = opts['T']
    if 'beta' in opts:
        beta = opts['beta']

    dim = np.size(X_train, 1)
    if np.size(lb) == 1:
        ub = ub * np.ones([1, dim], dtype='float')
        lb = lb * np.ones([1, dim], dtype='float')

    X = init_position(lb, ub, N, dim)

    fit = np.zeros([N, 1], dtype='float')
    Xrb = np.zeros([1, dim], dtype='float')
    fitR = float('inf')

    curve = np.zeros([1, max_iter], dtype='float')
    t = 0

    while t < max_iter:
        Xbin = binary_conversion(X, thres, N, dim)

        for i in range(N):
            fit[i, 0] = Fun(X_train, y_train, X_test, y_test, Xbin[i, :], opts)
            if fit[i, 0] < fitR:
                Xrb[0, :] = X[i, :]
                fitR = fit[i, 0]

        curve[0, t] = fitR.copy()
        print("*********************************", "Current Iteration Times:", t + 1, "***************************************")
        print("Best fitness value: ", curve[0, t])
        t += 1

        X_mu = np.zeros([1, dim], dtype='float')
        X_mu[0, :] = np.mean(X, axis=0)

        for i in range(N):
            E0 = -1 + 2 * rand()
            E = 2 * E0 * (1 - (t / max_iter))

            if abs(E) >= 1:
                q = rand()
                if q >= 0.5:
                    k = np.random.randint(low=0, high=N)
                    r1 = rand()
                    r2 = rand()
                    for d in range(dim):
                        X[i, d] = X[k, d] - r1 * abs(X[k, d] - 2 * r2 * X[i, d])
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif q < 0.5:
                    r3 = rand()
                    r4 = rand()
                    for d in range(dim):
                        X[i, d] = (Xrb[0, d] - X_mu[0, d]) - r3 * (lb[0, d] + r4 * (ub[0, d] - lb[0, d]))
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])


            elif abs(E) < 1:
                J = 2 * (1 - rand())
                r = rand()

                if r >= 0.5 and abs(E) >= 0.5:
                    for d in range(dim):
                        DX = Xrb[0, d] - X[i, d]
                        X[i, d] = DX - E * abs(J * Xrb[0, d] - X[i, d])
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])

                elif r >= 0.5 and abs(E) < 0.5:
                    for d in range(dim):
                        DX = Xrb[0, d] - X[i, d]
                        X[i, d] = Xrb[0, d] - E * abs(DX)
                        X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])


                elif r < 0.5 and abs(E) >= 0.5:
                    LF = levy_distribution(beta, dim)
                    Y = np.zeros([1, dim], dtype='float')
                    Z = np.zeros([1, dim], dtype='float')

                    for d in range(dim):

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X[i, d])

                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                    for d in range(dim):

                        Z[0, d] = Y[0, d] + rand() * LF[d]

                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)

                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY
                        X[i, :] = Y[0, :]
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ
                        X[i, :] = Z[0, :]

                elif r < 0.5 and abs(E) < 0.5:
                    # Levy distribution (9)
                    LF = levy_distribution(beta, dim)
                    Y = np.zeros([1, dim], dtype='float')
                    Z = np.zeros([1, dim], dtype='float')

                    for d in range(dim):

                        Y[0, d] = Xrb[0, d] - E * abs(J * Xrb[0, d] - X_mu[0, d])
                        Y[0, d] = boundary(Y[0, d], lb[0, d], ub[0, d])

                    for d in range(dim):

                        Z[0, d] = Y[0, d] + rand() * LF[d]
                        Z[0, d] = boundary(Z[0, d], lb[0, d], ub[0, d])

                    Ybin = binary_conversion(Y, thres, 1, dim)
                    Zbin = binary_conversion(Z, thres, 1, dim)
                    fitY = Fun(X_train, y_train, X_test, y_test, Ybin[0, :], opts)
                    fitZ = Fun(X_train, y_train, X_test, y_test, Zbin[0, :], opts)

                    if fitY < fit[i, 0]:
                        fit[i, 0] = fitY
                        X[i, :] = Y[0, :]
                    if fitZ < fit[i, 0]:
                        fit[i, 0] = fitZ
                        X[i, :] = Z[0, :]

    return X


if __name__ == '__main__':

    df = pd.read_csv(r'MHW.csv')
    y = df['NPP']
    X = df.drop('NPP', axis=1)
#    X = df.drop(['NPP', 'lat', 'lon'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    N = 10
    T = 2

    opts = {'N': N, 'T': T}

    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:
        best_max_depth = int(abs(fmdl[0][0])) + 8
    else:
        best_max_depth = int(abs(fmdl[0][0])) + 10

    if abs(fmdl[0][1]) > 0:
        best_n_estimators = int(abs(fmdl[0][1]))+200
    else:
        best_n_estimators = int(abs(fmdl[0][1]))+500

    print('----------------HRF Regression-Optimum Results-----------------')
    print("The best max_depth is " + str(abs(best_max_depth)))
    print("The best n_estimators is " + str(abs(best_n_estimators)))
    rfr_model = RandomForestRegressor(max_depth=best_max_depth, n_estimators=best_n_estimators)
    rfr_model.fit(X_train, y_train)
    y_pred = rfr_model.predict(X_test)
    accuracies = cross_val_score(rfr_model, X=X_train, y=y_train, cv=10)
    accuracy_mean = accuracies.mean()

    print('----------------Model evaluation-----------------')
    print('XGBoost Regression-Best Parameter-R2 Train:', evaluate_model(rfr_model, X_train, y_train))
    print('XGBoost Regression-Best Parameter-R2 Test:', evaluate_model(rfr_model, X_test, y_test))
    print('XGBoost Regression-Best Parameter-R2 Train: {}'.format(round(rfr_model.score(X_train, y_train), 5)))
    print('XGBoost Regression-Best Parameter-R2 Test: {}'.format(round(metrics.r2_score(y_test, y_pred), 5)))
    print('XGBoost Regression-Best Parameter-MSE: {}'.format(round(metrics.mean_squared_error(y_test, y_pred), 5)))
    print('XGBoost Regression-Best Parameter-Interpretable R2: {}'.format(round(metrics.explained_variance_score(y_test, y_pred), 5)))
    print('XGBoost Regression-Best Parameter-MAE: {}'.format(round(metrics.mean_absolute_error(y_test, y_pred), 5)))

    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")
    plt.legend(['Real', 'Prediction'])
    plt.title("Comparison of real value and prediction value based on HXGB")
    plt.show()

