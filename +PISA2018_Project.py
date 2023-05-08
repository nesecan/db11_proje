# !pip install missingno
# !pip install statsmodels pandas numpy
# pip install pyreadstat

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import pymc3 as pm
import statsmodels.api as sm
import statsmodels.formula.api as smf
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from catboost import CatBoostRegressor
# from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 100)


dfstudents = pd.read_excel("01.student data_revised.xlsx")
dfstudents.head()

# dfschools = pd.read_excel("PISA2018/01.school data_tur_revise copy 2.xlsx")
# dfschools.head()

def check_dfstudents(dfstudents, head=5):
    print("##################### Shape #####################")
    print(dfstudents.shape)
    print("##################### Types #####################")
    print(dfstudents.dtypes)
    print("##################### Head #####################")
    print(dfstudents.head(head))
    print("##################### Tail #####################")
    print(dfstudents.tail(head))
    print("##################### NA #####################")
    print(dfstudents.isnull().sum())
    print("##################### Quantiles #####################")
   # print(dfstudents.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_dfstudents(dfstudents)


dfstudents.dtypes
dfstudents ["CNTSCHID"]= dfstudents["CNTSCHID"].astype("object")
dfstudents ["ISCEDD"]= dfstudents["ISCEDD"].astype("object")
dfstudents ["ISCEDO"]= dfstudents["ISCEDO"].astype("object")
dfstudents ["REPEAT"]= dfstudents["REPEAT"].astype("object")

dfstudents ["BMMJ1"]= dfstudents["BMMJ1"].astype("object")
dfstudents ["BFMJ2"]= dfstudents["BFMJ2"].astype("object")
dfstudents ["DURECEC"]= dfstudents["DURECEC"].astype("object")
dfstudents ["BSMJ"]= dfstudents["BSMJ"].astype("object")
dfstudents ["GRADE_ST001D01T"]= dfstudents["GRADE_ST001D01T"].astype("object")
dfstudents ["GENDER"]= dfstudents["GENDER"].astype("object")


# dfstudents.drop("PROGN", axis=1, inplace=True)

################################################
# Numerik ve Kategorik Değişkenlerin Yakalanması
################################################

def grab_col_names(dfstudents, cat_th=10, car_th=20):
    cat_cols = [col for col in dfstudents.columns if dfstudents[col].dtypes == "O"]
    num_but_cat = [col for col in dfstudents.columns if dfstudents[col].nunique() < cat_th and
                   dfstudents[col].dtypes != "O"]
    cat_but_car = [col for col in dfstudents.columns if dfstudents[col].nunique() > car_th and
                   dfstudents[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dfstudents.columns if dfstudents[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dfstudents.shape[0]}")
    print(f"Variables: {dfstudents.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, cat_but_car, num_cols

cat_cols, cat_but_car, num_cols = grab_col_names(dfstudents)
# Observations: 6890
# Variables: 94
# cat_cols: 14
# num_cols: 75
# cat_but_car: 5
# num_but_cat: 8

# cat_cols + num_cols + cat_but_car = değişken sayısı.
# num_but_cat cat_cols'un içerisinde!


######################################
# Kategorik Değişken Analizi
######################################

def cat_summary(dfstudents, col_name, plot=False):
    print(pd.DataFrame({col_name: dfstudents[col_name].value_counts(),
                        "Ratio": 100 * dfstudents[col_name].value_counts() / len(dfstudents)}))
    if plot:
        sns.countplot(x=dfstudents[col_name], data=dfstudents)
        plt.show()

for col in cat_cols:
    cat_summary(dfstudents, col)


##################################
# Numerik değişkenlerin analizi
##################################

def num_summary(dfstudents, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dfstudents[numerical_col].describe(quantiles).T)

    if plot:
        dfstudents[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

for col in num_cols:
    num_summary(dfstudents, col)

# CNTSCHID, CNTSTUID: numerik değil, kategorisi fazla diye numerik gözüküyor.

##################################
# Eksik değer analizi
##################################

dfstudents.isnull().sum()
# ESCS               35
# ICTHOME            67
# ICTSCH             70
# MISCED             29
# FISCED             29

dfstudents.shape # (6890, 94)

def missing_values_table(dfstudents, na_name=False):
    na_columns = [col for col in dfstudents.columns if dfstudents[col].isnull().sum() > 0]
    n_miss = dfstudents[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dfstudents[na_columns].isnull().sum() / dfstudents.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(dfstudents, na_name=True)
#          n_miss  ratio
# ICTSCH       70  1.020
# ICTHOME      67  0.970
# ESCS         35  0.510
# MISCED       29  0.420
# FISCED       29  0.420

dfstudents["ICTSCH"].fillna(dfstudents["ICTSCH"].median(), inplace=True)
dfstudents["ICTHOME"].fillna(dfstudents["ICTHOME"].median(), inplace=True)
dfstudents["ESCS"].fillna(dfstudents["ESCS"].median(), inplace=True)
dfstudents["MISCED"].fillna(dfstudents["MISCED"].median(), inplace=True)
dfstudents["FISCED"].fillna(dfstudents["FISCED"].median(), inplace=True)

dfstudents.isnull().sum()
# hepsi 0, ama not applicable ve no response ve not applicable lar 95-99 gibi değerlerle doldurulmuş!

#not_applicable daki değişkenlerin hepsine bakıldı (95-99 gibi değerler için)
# DURECEC - 2890 tane 99, bunun dışındakilerin frekansı az.
dfstudents.drop("DURECEC", axis=1, inplace=True)

# dfstudents["GENDER"].unique()

not_applicable = ["ICTHOME", "ICTSCH", "CULTPOSS", "HEDRES", "WEALTH", "ICTRES", "DIRINS",
                  "PERFEED", "EMOSUPS", "ADAPTIVITY", "TEACHINT", "PISADIFF", "PERCOMP",
                  "PERCOOP", "ATTLNACT", "COMPETE", "WORKMAST", "GFOFAIL", "EUDMO", "SWBP",
                  "RESILIENCE", "MASTGOAL", "GCSELFEFF", "GCAWARE", "COGFLEX", "DISCRIM", "BELONG"]

dfstudents[not_applicable].describe().T

#               count   mean    std    min    25%    50%    75%    max
# ICTHOME    6890.000  6.633  6.751  0.000  4.000  6.000  8.000 99.000
# ICTSCH     6890.000  6.506 10.522  0.000  4.000  6.000  7.000 99.000
# CULTPOSS   6890.000  0.981 13.158 -2.747 -1.630 -0.817  0.027 99.000
# HEDRES     6890.000  0.544 10.101 -4.491 -1.310 -0.685  0.048 99.000
# WEALTH     6890.000 -0.817  7.398 -5.091 -1.968 -1.315 -0.742 99.000
# ICTRES     6890.000 -0.321  8.799 -3.814 -1.770 -0.940 -0.392 99.000
# DIRINS     6890.000  1.202  9.816 -2.942 -0.506  0.085  1.007 99.000
# PERFEED    6890.000  1.804 13.198 -1.639 -0.696 -0.036  0.774 99.000
# EMOSUPS    6890.000  3.104 17.244 -2.447 -0.658  0.503  1.035 99.000
# ADAPTIVITY 6890.000  1.704 12.658 -2.265 -0.579  0.107  0.546 99.000
# TEACHINT   6890.000  0.953 10.204 -2.218 -0.869  0.174  0.636 99.000
# PISADIFF   6890.000  2.365 14.447 -1.272 -0.402  0.271  0.917 99.000
# PERCOMP    6890.000  4.438 19.709 -1.989 -0.614  0.691  1.147 99.000
# PERCOOP    6890.000  4.631 20.961 -2.143 -0.939  0.601  0.867 99.000
# ATTLNACT   6890.000  1.685 13.272 -2.538 -0.658  0.028  1.084 99.000
# COMPETE    6890.000  2.026 12.913 -2.345 -0.411  0.196  1.264 99.000
# WORKMAST   6890.000  2.476 15.437 -2.736 -0.723 -0.102  0.998 99.000
# GFOFAIL    6890.000  2.200 14.229 -1.894 -0.531  0.110  0.821 99.000
# EUDMO      6890.000  2.173 14.032 -2.146 -0.672  0.262  0.927 99.000
# SWBP       6890.000  2.607 16.662 -3.067 -0.787 -0.593  0.730 99.000
# RESILIENCE 6890.000  1.798 11.911 -3.167 -0.187 -0.014  1.150 99.000
# MASTGOAL   6890.000  2.634 16.135 -2.525 -0.844 -0.338  0.835 99.000
# GCSELFEFF  6890.000  2.149 14.390 -2.714 -0.537  0.136  0.661 99.000
# GCAWARE    6890.000  2.346 14.698 -3.494 -0.444 -0.055  0.732 99.000
# COGFLEX    6890.000  2.551 15.093 -3.278 -0.449  0.166  0.738 99.000
# DISCRIM    6890.000  3.465 17.259 -1.155 -0.390  0.588  1.125 99.000
# BELONG     6890.000  1.339 12.075 -3.257 -0.769 -0.318  0.330 99.000


# Medianla doldurduk 95-99 ve 995-999 olanları
medians = dfstudents[not_applicable].median()
dfstudents.loc[:, not_applicable] = dfstudents.loc[:, not_applicable].apply(lambda x: x.mask(x > 95, medians[x.name]))
dfstudents[not_applicable].mean()

#MEDIAN ile replace ettikten sonra
# ICTHOME       6.215
# ICTSCH        5.399
# CULTPOSS     -0.772
# HEDRES       -0.483
# WEALTH       -1.356
# ICTRES       -1.090
# DIRINS        0.226
# PERFEED       0.021
# EMOSUPS       0.030
# ADAPTIVITY    0.067
# TEACHINT     -0.094
# PISADIFF      0.216
# PERCOMP       0.357
# PERCOOP       0.018
# ATTLNACT     -0.111
# COMPETE       0.320
# WORKMAST      0.017
# GFOFAIL       0.119
# EUDMO         0.152
# SWBP         -0.269
# RESILIENCE    0.347
# MASTGOAL     -0.062
# GCSELFEFF     0.025
# GCAWARE       0.118
# COGFLEX       0.198
# DISCRIM       0.366
# BELONG       -0.145


##################################
# Aykırı değer analizi
##################################

def outlier_thresholds(dfstudents, col_name, q1=0.05, q3=0.95):
    quartile1 = dfstudents[col_name].quantile(q1)
    quartile3 = dfstudents[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dfstudents, col_name):
    low_limit, up_limit = outlier_thresholds(dfstudents, col_name)
    if dfstudents[(dfstudents[col_name] > up_limit) | (dfstudents[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dfstudents, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dfstudents, variable, q1=0.05, q3=0.95)
    dfstudents.loc[(dfstudents[variable] < low_limit), variable] = low_limit
    dfstudents.loc[(dfstudents[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(dfstudents, col))
    if check_outlier(dfstudents, col):
        replace_with_thresholds(dfstudents, col)

# Hepsi FALSE döndü


##############################
# ENCODING
##############################

cat_cols, cat_but_car, num_cols = grab_col_names(dfstudents)
# Observations: 6890
# Variables: 93
# cat_cols: 13
# num_cols: 75
# cat_but_car: 5
# num_but_cat: 8

##############################################
# ENCODE ETMEYE GEREK YOK, hepsi numerik çünkü?

def label_encoder(dfstudents, ordinal_col):
    labelencoder = LabelEncoder()
    dfstudents[ordinal_col] = labelencoder.fit_transform(dfstudents[ordinal_col])
    return dfstudents

# ordinal_cols = dfstudents[["GRADE_ST001D01T", "ISCEDL", "MISCED", "FISCED", "HISCED", "PAREDINT",
# "BMMJ1", "BFMJ2", "IMMIG", "BSMJ", "STRATUM"]]

ordinal_cols = dfstudents[["STRATUM"]]

for col in ordinal_cols:
    label_encoder(dfstudents, col)

dfstudents.head()

# dfstudents ["PROGN"]= dfstudents["PROGN"].astype("object")
# dfstudents ["LANGN"]= dfstudents["LANGN"].astype("object")


dfstudents.describe().T
# PVSCIE skorları normal dağılıyor gibi. Onun için standartScaler kullanabiliriz.

scaler = StandardScaler()
dfstudents[num_cols] = scaler.fit_transform(dfstudents[num_cols])
dfstudents[num_cols].describe().T # z-scoreları elde ettik


##################################
# Korelasyon analizi
##################################

dfstudents[num_cols].corr()
# PVler arasında yüksek korelasyon var >.80!


##############
# PV1SCIE için;
##############
y1 = dfstudents["PV1SCIE"]
X1 = dfstudents.drop(["PV1SCIE", "CNTSTUID"], axis=1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20, random_state=17)
model1 = [("CART", DecisionTreeRegressor())]

for name, regressor in model1:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X1, y1, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5221 (CART)

##############
# PV2SCIE için;
##############
y2 = dfstudents["PV2SCIE"]
X2 = dfstudents.drop(["PV2SCIE", "CNTSTUID"], axis=1)

X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.20, random_state=17)
model2 = [("CART", DecisionTreeRegressor())]

for name, regressor in model2:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X2, y2, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
#RMSE: 0.522 (CART)

##############
# PV3SCIE için;
##############
y3 = dfstudents["PV3SCIE"]
X3 = dfstudents.drop(["PV3SCIE", "CNTSTUID"], axis=1)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.20, random_state=17)
model3 = [("CART", DecisionTreeRegressor())]

for name, regressor in model3:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X3, y3, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5147 (CART)

##############
# PV4SCIE için;
##############
y4 = dfstudents["PV4SCIE"]
X4 = dfstudents.drop(["PV4SCIE", "CNTSTUID"], axis=1)

X4_train, X4_test, y4_train, y4_test = train_test_split(X4, y4, test_size=0.20, random_state=17)
model4 = [("CART", DecisionTreeRegressor())]

for name, regressor in model4:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X4, y4, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5347 (CART)

##############
# PV5SCIE için;
##############
y5 = dfstudents["PV5SCIE"]
X5 = dfstudents.drop(["PV5SCIE", "CNTSTUID"], axis=1)

X5_train, X5_test, y5_train, y5_test = train_test_split(X5, y5, test_size=0.20, random_state=17)
model5 = [("CART", DecisionTreeRegressor())]

for name, regressor in model5:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X5, y5, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5153 (CART)

##############
# PV6SCIE için;
##############
y6 = dfstudents["PV6SCIE"]
X6 = dfstudents.drop(["PV6SCIE", "CNTSTUID"], axis=1)

X6_train, X6_test, y6_train, y6_test = train_test_split(X6, y6, test_size=0.20, random_state=17)
model6 = [("CART", DecisionTreeRegressor())]

for name, regressor in model6:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X6, y6, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5179 (CART)

##############
# PV7SCIE için;
##############
y7 = dfstudents["PV7SCIE"]
X7 = dfstudents.drop(["PV7SCIE", "CNTSTUID"], axis=1)

X7_train, X7_test, y7_train, y7_test = train_test_split(X7, y7, test_size=0.20, random_state=17)
model7 = [("CART", DecisionTreeRegressor())]

for name, regressor in model7:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X7, y7, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5189 (CART)

##############
# PV8SCIE için;
##############
y8 = dfstudents["PV8SCIE"]
X8 = dfstudents.drop(["PV8SCIE", "CNTSTUID"], axis=1)

X8_train, X8_test, y8_train, y8_test = train_test_split(X8, y8, test_size=0.20, random_state=17)
model8 = [("CART", DecisionTreeRegressor())]

for name, regressor in model8:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X8, y8, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5275 (CART)

##############
# PV9SCIE için;
##############
y9 = dfstudents["PV9SCIE"]
X9 = dfstudents.drop(["PV9SCIE", "CNTSTUID"], axis=1)

X9_train, X9_test, y9_train, y9_test = train_test_split(X9, y9, test_size=0.20, random_state=17)
model9 = [("CART", DecisionTreeRegressor())]

for name, regressor in model9:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X9, y9, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5218 (CART)

##############
# PV10SCIE için;
##############
y10 = dfstudents["PV10SCIE"]
X10 = dfstudents.drop(["PV10SCIE", "CNTSTUID"], axis=1)

X10_train, X10_test, y10_train, y10_test = train_test_split(X10, y10, test_size=0.20, random_state=17)
model10 = [("CART", DecisionTreeRegressor())]

for name, regressor in model10:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X10, y10, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name})")
# RMSE: 0.5200 (CART)


# RMSE (max): PV4SCIE
# RMSE (min): PV3SCIE 0.5147 - Bu value yu seçtik DV olarak


#############################################################################################