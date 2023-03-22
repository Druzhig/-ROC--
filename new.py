import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_excel(r'C:\Users\hp\Desktop\data science\bankloan.xlsx')
df = df.dropna()
X_train, X_test, Y_train, Y_test = train_test_split(df.drop('default', axis=1), df['default'], test_size=0.2)

df['income'] = np.log(df['income'] + 1)
df['debtinc'] = np.log(df['debtinc'] + 1)
df['creddebt'] = np.log(df['creddebt'] + 1)
df['address'] = np.log(df["address"] + 1)
df['employ'] = np.log(df['employ'] + 1)
df['othdebt'] = np.log(df['othdebt'] + 1)

df.corr().style.background_gradient(cmap="coolwarm")
factors = []
cor_max = 0.6
target = 'default'
df = df[[target] + [x for x in df.columns if x != target]]
cor_matrix = df.corr()
for i in range(1, len(cor_matrix) - 1):
    for j in range(i + 1, len(cor_matrix)):
        if np.abs(cor_matrix.iloc[i, j]) >= cor_max:
            if abs(cor_matrix.iloc[0, j]) > abs(cor_matrix.iloc[0, i]):
                factors.append(i)
            else:
                factors.append(j)
names_to_del = cor_matrix.columns[list(set(factors))]
df = df.drop(names_to_del, axis=1)
model = sm.Logit(Y_train, sm.add_constant(X_train)).fit()
p_max = 0.05
while np.max(model.pvalues[1:]) > p_max:
    col = X_train.columns[np.argmax(model.pvalues[1:])]
    X_train = X_train.drop(col, axis=1)
    X_test = X_test.drop(col, axis=1)
    model = sm.Logit(Y_train, sm.add_constant(X_train)).fit()

predict = model.predict(sm.add_constant(X_test))
fpr, tpr, thr = roc_curve(Y_test, predict)
logit_auc = roc_auc_score(Y_test, predict)
plt.plot(fpr, tpr, label= f"Logit ROC =  {logit_auc}")
plt.plot()
plt.legend()
plt.plot([0, 1], [0, 1], color = 'red')
plt.show()
