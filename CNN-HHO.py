import numpy as np
from numpy.random import rand
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.utils import plot_model
import keras.layers as layers
import keras.backend as K
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_error, r2_score
import seaborn as sns



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
        units = int(abs(x[0])) * 10
    else:
        units = int(abs(x[0])) + 16

    if abs(x[1]) > 0:
        epochs = int(abs(x[1])) * 10
    else:
        epochs = int(abs(x[1])) + 10


    cnn_model = Sequential()
    cnn_model.add(
        Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu',
               padding='valid'))
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=units, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])
    cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=64)
    score = cnn_model.evaluate(X_test, y_test, batch_size=128)


    fitness_value = (1 - float(score[1]))

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
    lb = 0
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
            E0 = -1 + 2 * rand()  #
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
    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)

    print('***********************Shape of training dataset**************************')
    print(X_train.shape)

    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)
    print('***********************Shape of test dataset**************************')
    print(X_test.shape)

    N = 10
    T = 2

    opts = {'N': N, 'T': T}

    fmdl = jfs(X_train, y_train, X_test, y_test, opts)

    if abs(fmdl[0][0]) > 0:
        best_units = int(abs(fmdl[0][0])) * 10 + 48
    else:
        best_units = int(abs(fmdl[0][0])) + 48

    if abs(fmdl[0][1]) > 0:
        best_epochs = int(abs(fmdl[0][1])) * 10 + 60
    else:
        best_epochs = (int(abs(fmdl[0][1])) + 100)

    print('----------------HCNN Regression-Optimum Results-----------------')
    print("The best units is " + str(abs(best_units)))
    print("The best epochs is " + str(abs(best_epochs)))

    cnn_model = Sequential()
    cnn_model.add(Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1),
                         activation='relu'))
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(units=best_units, activation='relu'))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Dense(1))
    cnn_model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['mse'])

    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=best_epochs, batch_size=64)  # 拟合

    print('*************************Brief info of model*******************************')
    print(cnn_model.summary())

    plot_model(cnn_model, to_file='model.png', show_shapes=True)


    def show_history(history):
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, 'r', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Test loss')
        plt.title('Training and Test loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()


    show_history(history)

    y_pred = cnn_model.predict(X_test, batch_size=10)

    print('----------------Model evaluation-----------------')
    print('CNN Regression-Best Parameter-R2 Train:：', cnn_model.evaluate(X_train, y_train))
    print('CNN Regression-Best Parameter-R2 Test:', round(r2_score(y_test, y_pred), 5))
    print('CNN Regression-Best Parameter-MSE: {}', round(mean_squared_error(y_test, y_pred), 5))
    print('CNN Regression-Best Parameter-Interpretable R2: {}', round(explained_variance_score(y_test, y_pred), 5))
    print('CNN Regression-Best Parameter-MAE: {}', round(mean_absolute_error(y_test, y_pred), 5))

    # 真实值与预测值比对图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(len(y_test)), y_test, color="blue", linewidth=1.5, linestyle="-")
    plt.plot(range(len(y_pred)), y_pred, color="red", linewidth=1.5, linestyle="-.")
    plt.legend(['Real', 'Prediction'])  # 设置图例
    plt.title("Comparison of real value and prediction value based on HCNN")
    plt.show()
