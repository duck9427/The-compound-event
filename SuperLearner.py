import numpy as np
import pandas as pd
from mlens.ensemble import SuperLearner
from mlens.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import XGB-HHO as XGBRegression
import RF-HHO as RandomForestRegression
import CNN-HHO as CNNRegression

# import data
df = pd.read_csv(r'Data.csv')
X = df.drop('NPP', axis=1)
y = df['NPP']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base learners
RF = RandomForestRegression()
XGB = LinearRegression()
CNN = CNNRegression()

# Define scorer
def evaluate_model(model, x, y):
    pred = model.predict(x)
    r2 = r2_score(y, pred)
    corr_coef = np.corrcoef(y, pred)[0, 1]
    return r2, corr_coef

# define base learners
base_learners = [('RF', RandomForestRegression()),
                 ('XGB', XGBRegression()),
                 ('CNN', CNNRegression())]

# define meta learner
meta_learner = XGBRegression()

# define scorer
scorer = make_scorer(r2_score)

# create super learner ensemble
ensemble = SuperLearner(scorer=scorer,
                        folds=5,
                        shuffle=True,
                        random_state=42,
                        verbose=2,
                        backend="multiprocessing")

# add base learners to ensemble
ensemble.add(list(map(lambda x: x[1], base_learners)))

# add meta learner to ensemble
ensemble.add_meta(meta_learner)

# fit ensemble on training data
ensemble.fit(X_train, y_train)

# predict on test data
y_pred = ensemble.predict(X_test)

# print accuracy score
print("R2:", r2_score(y_test, y_pred))