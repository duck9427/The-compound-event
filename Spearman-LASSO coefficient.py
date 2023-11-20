from numpy import sqrt
from sklearn import metrics


def data_explore():
    import numpy as np
    import pandas as pd

    inputfile = r'data'
    data = pd.read_csv(inputfile)

    description = [data.min(), data.max(), data.mean(), data.std()]
    description = pd.DataFrame(description, index=['Min', 'Max', 'Mean', 'STD']).T
    print('Description: \n', np.round(description, 2))
    corr = data.corr(method='pearson')
    print('Correlation matrix: \n', np.round(corr, 2))


    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.subplots(figsize=(10, 10))
    plt.rcParams['axes.unicode_minus'] = False
    sns.heatmap(corr, annot=True, vmax=1, square=True, cmap="Blues")
    plt.title('Correlation heatmap')
    plt.show()


def data_preprocessed():
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import Lasso

    inputfile = r'data'
    data = pd.read_csv(inputfile)
    lasso = Lasso(0.1)
    y = data['S']
    X = data.drop('S', axis=1)
    lasso.fit(X, y)
    print('Coefficient: ', np.round(lasso.coef_, 5))
    lasso_coef=pd.DataFrame(np.round(lasso.coef_, 5))
    lasso_coef.to_csv(r'result')
    print('The number of non zero coefficient: ', np.sum(lasso.coef_ != 0))

    mask = lasso.coef_ != 0
    print('Whether coefficient is zero or not:', mask)

#    outputfile = 'D:\Kenkyu\Research\Origin画图\Comparison\enddata8.csv'
#    print('************************')
#    print(data.head())
#    new_reg_data = data.loc[:, ['Temperature', 'Fe', 'O2', 'Silicate', 'Nitrate', 'pH', 'Phosphate', 'Salinity', 'SPCO2', 'Chla', 'PAR', 'DOC', 'NH4', 'NPP']]  # 返回相关系数非零的数据
#    new_reg_data.to_csv(outputfile, index=False)
#    print('输出数据的维度为：', new_reg_data.shape)


data_explore()
data_preprocessed()