import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from tqdm import tqdm
import random
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import heapq
import tslearn
from tslearn import metrics

# %matplotlib inline
# reading data
train_df1 = pd.read_csv("train_FD001.txt", sep=" ", header=None)
test_df1 = pd.read_csv("test_FD001.txt", sep=" ", header=None)
truth_df1 = pd.read_csv("RUL_FD001.txt", sep=" ", header=None)

train_df1.drop(train_df1.columns[[26, 27]], axis=1, inplace=True)
test_df1.drop(test_df1.columns[[26, 27]], axis=1, inplace=True)
truth_df1.drop(truth_df1.columns[[1]], axis=1, inplace=True)

data_index = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
              's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
              's15', 's16', 's17', 's18', 's19', 's20', 's21']

train_df1.columns = data_index
test_df1.columns = data_index

# attaching label
rul = pd.DataFrame(train_df1.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df1 = train_df1.merge(rul, on=['id'], how='left')

train_df1['RUL'] = train_df1['max'] - train_df1['cycle']
# train_df1.loc[train_df1['RUL'] >= 125, 'RUL'] = 125
train_df1.drop('max', axis=1, inplace=True)

# normalize training data
min_max_scaler = preprocessing.MinMaxScaler()
cols_normalize = train_df1.columns.difference(['id', 'cycle', 'RUL'])
norm_train_df1 = pd.DataFrame(min_max_scaler.fit_transform(train_df1[cols_normalize]),
                              columns=cols_normalize,
                              index=train_df1.index)
join_df = train_df1[train_df1.columns.difference(cols_normalize)].join(norm_train_df1)
train_df1 = join_df.reindex(columns=train_df1.columns)

# normalize testing data
cols_normalize = test_df1.columns.difference(['id', 'cycle'])
norm_test_df1 = pd.DataFrame(min_max_scaler.transform(test_df1[cols_normalize]),
                             columns=cols_normalize,
                             index=test_df1.index)
join_df = test_df1[test_df1.columns.difference(cols_normalize)].join(norm_test_df1)
test_df1 = join_df.reindex(columns=test_df1.columns)

# remove meaningless data
train_df1.drop(columns=['s1', 's10', 's16', 's18', 's19', 's5', 's6', 'setting1', 'setting2', 'setting3'], axis=1,
               inplace=True)
test_df1.drop(columns=['s1', 's10', 's16', 's18', 's19', 's5', 's6', 'setting1', 'setting2', 'setting3'], axis=1,
              inplace=True)

# Remove the tag column, leaving only useful monitoring data
train_df1_id1 = train_df1.loc[train_df1['id'] == 1]
train_df1_id1.drop(columns=train_df1_id1.columns.difference(['s2', 's3','s4','s7','s11','s12','s15','s20']),axis=1, inplace=True)  # s9与s14高度相关，去掉s9
fig = plt.figure(figsize=(9,9))
ax = fig.gca()
train_df1_id1.hist(ax=ax)
plt.tight_layout()
plt.show()

# PCA is used to reduce the dimension of 14 columns of monitoring data to 1 dimension and visualize it,training data
pca = PCA(n_components=3)
train_df1_pca = train_df1.drop(columns=train_df1.columns.difference(['s2', 's3','s4','s7','s11','s12','s15','s20']),axis=1, inplace=False)
train_df1_pca_result = pca.fit_transform(train_df1_pca.values)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# PCA is used to reduce the dimension of 14 columns of monitoring data to 1 dimension and visualize it,testing data
test_df1_pca = test_df1.drop(columns=test_df1.columns.difference(['s2', 's3','s4','s7','s11','s12','s15','s20']),axis=1, inplace=False)
test_df1_pca_result = pca.transform(test_df1_pca.values)

# Data smoothing
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

# KRR smoothing and visualization of training data
train_df1_smooth = []

kr = KernelRidge(kernel='rbf', gamma=0.01, alpha=0.1)
for i in range(train_df1['id'].max()):
    mask = train_df1['id'] == i + 1
    X = np.array(train_df1['cycle'][mask]).reshape(-1, 1)
    y = train_df1_pca_result[:, 0][mask]
    kr.fit(X, y)
    y_kr = kr.predict(X)
    train_df1_smooth.append(y_kr)

# KRR smoothing and visualization of testing data
test_df1_smooth = []
for i in range(test_df1['id'].max()):
    mask = test_df1['id'] == i + 1
    X = np.array(test_df1['cycle'][mask]).reshape(-1, 1)
    y = test_df1_pca_result[:, 0][mask]

    kr.fit(X, y)
    y_kr = kr.predict(X)
    test_df1_smooth.append(y_kr)

# Connect the smoothed data together
train_df1_smooth_te = np.vstack((train_df1_smooth[0].reshape(-1, 1), train_df1_smooth[1].reshape(-1, 1), train_df1_smooth[2].reshape(-1, 1), train_df1_smooth[3].reshape(-1, 1), train_df1_smooth[4].reshape(-1, 1),
                 train_df1_smooth[5].reshape(-1, 1), train_df1_smooth[6].reshape(-1, 1), train_df1_smooth[7].reshape(-1, 1), train_df1_smooth[8].reshape(-1, 1), train_df1_smooth[9].reshape(-1, 1),
                 train_df1_smooth[10].reshape(-1, 1), train_df1_smooth[11].reshape(-1, 1), train_df1_smooth[12].reshape(-1, 1), train_df1_smooth[13].reshape(-1, 1), train_df1_smooth[14].reshape(-1, 1),
                 train_df1_smooth[15].reshape(-1, 1), train_df1_smooth[16].reshape(-1, 1), train_df1_smooth[17].reshape(-1, 1), train_df1_smooth[18].reshape(-1, 1), train_df1_smooth[19].reshape(-1, 1),
                 train_df1_smooth[20].reshape(-1, 1), train_df1_smooth[21].reshape(-1, 1), train_df1_smooth[22].reshape(-1, 1), train_df1_smooth[23].reshape(-1, 1), train_df1_smooth[24].reshape(-1, 1),
                 train_df1_smooth[25].reshape(-1, 1), train_df1_smooth[26].reshape(-1, 1), train_df1_smooth[27].reshape(-1, 1), train_df1_smooth[28].reshape(-1, 1), train_df1_smooth[29].reshape(-1, 1),
                 train_df1_smooth[30].reshape(-1, 1), train_df1_smooth[31].reshape(-1, 1), train_df1_smooth[32].reshape(-1, 1), train_df1_smooth[33].reshape(-1, 1), train_df1_smooth[34].reshape(-1, 1),
                 train_df1_smooth[35].reshape(-1, 1), train_df1_smooth[36].reshape(-1, 1), train_df1_smooth[37].reshape(-1, 1), train_df1_smooth[38].reshape(-1, 1), train_df1_smooth[39].reshape(-1, 1),
                 train_df1_smooth[40].reshape(-1, 1), train_df1_smooth[41].reshape(-1, 1), train_df1_smooth[42].reshape(-1, 1), train_df1_smooth[43].reshape(-1, 1), train_df1_smooth[44].reshape(-1, 1),
                 train_df1_smooth[45].reshape(-1, 1), train_df1_smooth[46].reshape(-1, 1), train_df1_smooth[47].reshape(-1, 1), train_df1_smooth[48].reshape(-1, 1), train_df1_smooth[49].reshape(-1, 1),
                 train_df1_smooth[50].reshape(-1, 1), train_df1_smooth[51].reshape(-1, 1), train_df1_smooth[52].reshape(-1, 1), train_df1_smooth[53].reshape(-1, 1), train_df1_smooth[54].reshape(-1, 1),
                 train_df1_smooth[55].reshape(-1, 1), train_df1_smooth[56].reshape(-1, 1), train_df1_smooth[57].reshape(-1, 1), train_df1_smooth[58].reshape(-1, 1), train_df1_smooth[59].reshape(-1, 1),
                 train_df1_smooth[60].reshape(-1, 1), train_df1_smooth[61].reshape(-1, 1), train_df1_smooth[62].reshape(-1, 1), train_df1_smooth[63].reshape(-1, 1), train_df1_smooth[64].reshape(-1, 1),
                 train_df1_smooth[65].reshape(-1, 1), train_df1_smooth[66].reshape(-1, 1), train_df1_smooth[67].reshape(-1, 1), train_df1_smooth[68].reshape(-1, 1), train_df1_smooth[69].reshape(-1, 1),
                 train_df1_smooth[70].reshape(-1, 1), train_df1_smooth[71].reshape(-1, 1), train_df1_smooth[72].reshape(-1, 1), train_df1_smooth[73].reshape(-1, 1), train_df1_smooth[74].reshape(-1, 1),
                 train_df1_smooth[75].reshape(-1, 1), train_df1_smooth[76].reshape(-1, 1), train_df1_smooth[77].reshape(-1, 1), train_df1_smooth[78].reshape(-1, 1), train_df1_smooth[79].reshape(-1, 1),
                 train_df1_smooth[80].reshape(-1, 1), train_df1_smooth[81].reshape(-1, 1), train_df1_smooth[82].reshape(-1, 1), train_df1_smooth[83].reshape(-1, 1), train_df1_smooth[84].reshape(-1, 1),
                 train_df1_smooth[85].reshape(-1, 1), train_df1_smooth[86].reshape(-1, 1), train_df1_smooth[87].reshape(-1, 1), train_df1_smooth[88].reshape(-1, 1), train_df1_smooth[89].reshape(-1, 1),
                 train_df1_smooth[90].reshape(-1, 1), train_df1_smooth[91].reshape(-1, 1), train_df1_smooth[92].reshape(-1, 1), train_df1_smooth[93].reshape(-1, 1), train_df1_smooth[94].reshape(-1, 1),
                 train_df1_smooth[95].reshape(-1, 1), train_df1_smooth[96].reshape(-1, 1), train_df1_smooth[97].reshape(-1, 1), train_df1_smooth[98].reshape(-1, 1), train_df1_smooth[99].reshape(-1, 1)))
test_df1_smooth_te = np.vstack((test_df1_smooth[0].reshape(-1, 1), test_df1_smooth[1].reshape(-1, 1), test_df1_smooth[2].reshape(-1, 1), test_df1_smooth[3].reshape(-1, 1), test_df1_smooth[4].reshape(-1, 1),
                 test_df1_smooth[5].reshape(-1, 1), test_df1_smooth[6].reshape(-1, 1), test_df1_smooth[7].reshape(-1, 1), test_df1_smooth[8].reshape(-1, 1), test_df1_smooth[9].reshape(-1, 1),
                 test_df1_smooth[10].reshape(-1, 1), test_df1_smooth[11].reshape(-1, 1), test_df1_smooth[12].reshape(-1, 1), test_df1_smooth[13].reshape(-1, 1), test_df1_smooth[14].reshape(-1, 1),
                 test_df1_smooth[15].reshape(-1, 1), test_df1_smooth[16].reshape(-1, 1), test_df1_smooth[17].reshape(-1, 1), test_df1_smooth[18].reshape(-1, 1), test_df1_smooth[19].reshape(-1, 1),
                 test_df1_smooth[20].reshape(-1, 1), test_df1_smooth[21].reshape(-1, 1), test_df1_smooth[22].reshape(-1, 1), test_df1_smooth[23].reshape(-1, 1), test_df1_smooth[24].reshape(-1, 1),
                 test_df1_smooth[25].reshape(-1, 1), test_df1_smooth[26].reshape(-1, 1), test_df1_smooth[27].reshape(-1, 1), test_df1_smooth[28].reshape(-1, 1), test_df1_smooth[29].reshape(-1, 1),
                 test_df1_smooth[30].reshape(-1, 1), test_df1_smooth[31].reshape(-1, 1), test_df1_smooth[32].reshape(-1, 1), test_df1_smooth[33].reshape(-1, 1), test_df1_smooth[34].reshape(-1, 1),
                 test_df1_smooth[35].reshape(-1, 1), test_df1_smooth[36].reshape(-1, 1), test_df1_smooth[37].reshape(-1, 1), test_df1_smooth[38].reshape(-1, 1), test_df1_smooth[39].reshape(-1, 1),
                 test_df1_smooth[40].reshape(-1, 1), test_df1_smooth[41].reshape(-1, 1), test_df1_smooth[42].reshape(-1, 1), test_df1_smooth[43].reshape(-1, 1), test_df1_smooth[44].reshape(-1, 1),
                 test_df1_smooth[45].reshape(-1, 1), test_df1_smooth[46].reshape(-1, 1), test_df1_smooth[47].reshape(-1, 1), test_df1_smooth[48].reshape(-1, 1), test_df1_smooth[49].reshape(-1, 1),
                 test_df1_smooth[50].reshape(-1, 1), test_df1_smooth[51].reshape(-1, 1), test_df1_smooth[52].reshape(-1, 1), test_df1_smooth[53].reshape(-1, 1), test_df1_smooth[54].reshape(-1, 1),
                 test_df1_smooth[55].reshape(-1, 1), test_df1_smooth[56].reshape(-1, 1), test_df1_smooth[57].reshape(-1, 1), test_df1_smooth[58].reshape(-1, 1), test_df1_smooth[59].reshape(-1, 1),
                 test_df1_smooth[60].reshape(-1, 1), test_df1_smooth[61].reshape(-1, 1), test_df1_smooth[62].reshape(-1, 1), test_df1_smooth[63].reshape(-1, 1), test_df1_smooth[64].reshape(-1, 1),
                 test_df1_smooth[65].reshape(-1, 1), test_df1_smooth[66].reshape(-1, 1), test_df1_smooth[67].reshape(-1, 1), test_df1_smooth[68].reshape(-1, 1), test_df1_smooth[69].reshape(-1, 1),
                 test_df1_smooth[70].reshape(-1, 1), test_df1_smooth[71].reshape(-1, 1), test_df1_smooth[72].reshape(-1, 1), test_df1_smooth[73].reshape(-1, 1), test_df1_smooth[74].reshape(-1, 1),
                 test_df1_smooth[75].reshape(-1, 1), test_df1_smooth[76].reshape(-1, 1), test_df1_smooth[77].reshape(-1, 1), test_df1_smooth[78].reshape(-1, 1), test_df1_smooth[79].reshape(-1, 1),
                 test_df1_smooth[80].reshape(-1, 1), test_df1_smooth[81].reshape(-1, 1), test_df1_smooth[82].reshape(-1, 1), test_df1_smooth[83].reshape(-1, 1), test_df1_smooth[84].reshape(-1, 1),
                 test_df1_smooth[85].reshape(-1, 1), test_df1_smooth[86].reshape(-1, 1), test_df1_smooth[87].reshape(-1, 1), test_df1_smooth[88].reshape(-1, 1), test_df1_smooth[89].reshape(-1, 1),
                 test_df1_smooth[90].reshape(-1, 1), test_df1_smooth[91].reshape(-1, 1), test_df1_smooth[92].reshape(-1, 1), test_df1_smooth[93].reshape(-1, 1), test_df1_smooth[94].reshape(-1, 1),
                 test_df1_smooth[95].reshape(-1, 1), test_df1_smooth[96].reshape(-1, 1), test_df1_smooth[97].reshape(-1, 1), test_df1_smooth[98].reshape(-1, 1), test_df1_smooth[99].reshape(-1, 1)))


RUL_truth = []
for i in range(len(truth_df1)):
    RUL_truth_temp = truth_df1[0][i]
    if RUL_truth_temp > 124:
        RUL_truth_temp = 125
    RUL_truth.append(RUL_truth_temp)


# Dynamic Time Warping
def similarity(id_number, k3):
    min_distance = []
    import heapq

    for i in range(train_df1['id'].max()):
        mask1 = train_df1['id'] == i + 1
        mask2 = test_df1['id'] == id_number
        train_data = train_df1_smooth_te[mask1].flatten()
        test_data = test_df1_smooth_te[mask2].flatten()
        interval1 = np.arange(-k3, k3 + 1)
        distance_temp = []

        for j in range(len(interval1)):
            if (len(test_data)) >= (len(train_data) + k3):
                path, d = tslearn.metrics.dtw_path(train_data, test_data)
                distance_temp.append(d)
            else:
                if len(test_data) + interval1[j] <= len(train_data):
                    train_data_temp = train_data[:(len(test_data) + interval1[j])]
                    path, d = tslearn.metrics.dtw_path(train_data_temp, test_data)
                    distance_temp.append(d)

        min_distance_temp = min(distance_temp)
        min_distance.append(min_distance_temp)

    return min_distance


# outlier remove
def outlier_remove(id_number, top_numbers, id_min_distance):
    j = id_number  # engine id

    dist_min_id = []
    min_number = heapq.nsmallest(top_numbers, id_min_distance)
    for i in range(top_numbers):
        dist_min_id.append(id_min_distance.index(min_number[i]) + 1)

    dist_min = []
    RUL_temp = []

    # Delete those with a total life greater than 145 and less than 125
    id_to_delete = []
    for i in range(len(dist_min_id)):
        RUL_train = len(train_df1_smooth[dist_min_id[i] - 1].reshape(-1, 1))
        RUL_test = len(test_df1_smooth[j - 1].reshape(-1, 1))
        RUL = RUL_train - RUL_test
        if RUL >= 145:
            id_to_delete.append(dist_min_id[i])
        if RUL_train < 125:
            id_to_delete.append(dist_min_id[i])

    for i in range(len(id_to_delete)):
        dist_min_id.remove(id_to_delete[i])

    # Median removal
    for i in range(len(dist_min_id)):
        RUL_train = len(train_df1_smooth[dist_min_id[i] - 1].reshape(-1, 1))
        RUL_test = len(test_df1_smooth[j - 1].reshape(-1, 1))
        RUL = RUL_train - RUL_test
        RUL_temp.append(RUL)

    RUL_temp.sort()
    RUL_25 = np.median(RUL_temp[:len(RUL_temp) // 2])
    RUL_50 = np.median(RUL_temp)
    RUL_75 = np.median(RUL_temp[len(RUL_temp) // 2:])

    id_to_delete = []
    for i in range(len(dist_min_id)):
        if (RUL_50 - 3 * (RUL_50 - RUL_25)) > RUL_temp[i]:
            id_to_delete.append(dist_min_id[i])
        if (RUL_50 + 2 * (RUL_75 - RUL_50)) < RUL_temp[i]:
            id_to_delete.append(dist_min_id[i])

    for i in range(len(id_to_delete)):
        dist_min_id.remove(id_to_delete[i])

    for i in range(len(dist_min_id)):
        dist_min.append(id_min_distance[dist_min_id[i] - 1])

    return dist_min_id, dist_min

min_distance = []
for i in tqdm(range(100)):
    dist_min_all = similarity(id_number=i+1, k3=4)
    min_distance.append(dist_min_all)


def Simple_RUL_trajectory(id_number, top_numbers, id_min_distance, k):
    dist_min_id, dist_min = outlier_remove(id_number=id_number, top_numbers=top_numbers,
                                           id_min_distance=id_min_distance)
    RUL_train = []
    for i in range(len(dist_min_id)):
        RUL_train.append(len(train_df1_smooth[dist_min_id[i] - 1]) - len(test_df1_smooth[id_number - 1]))
    RUL = 0
    RUL = k * min(RUL_train) + (1 - k) * max(RUL_train)
    if RUL < 0:
        RUL = 0
    if RUL > 125:
        RUL = 125
    return RUL


def RUL_pred_cal(k1, k2):
    RUL = []
    for i in range(100):
        RUL_temp = Simple_RUL_trajectory(id_number=(i + 1), top_numbers=k2, id_min_distance=min_distance[i], k=k1)
        RUL.append(RUL_temp)

    RUL_pred = [round(i, 1) for i in RUL]

    return RUL_pred


def score_cal(RUL_pred, RUL_truth):
    x1 = np.array(RUL_pred).reshape(-1, 1)
    x2 = np.array(RUL_truth).reshape(-1, 1)
    d = np.array(x1 - x2)
    tmp = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        if d[i, 0] >= 0:
            tmp[i] = np.exp(d[i, 0] / 10) - 1
        else:
            tmp[i] = np.exp(-d[i, 0] / 13) - 1

    return sum(tmp)

RUL = []
for i in range(100):
    RUL_temp = Simple_RUL_trajectory(id_number = (i+1), top_numbers=7, id_min_distance=min_distance[i], k=0.385)
    RUL.append(RUL_temp)

RUL_pred = [round(i,1) for i in RUL]

from sklearn.metrics import confusion_matrix, recall_score, precision_score, mean_squared_error, mean_absolute_error
print('\nRMSE: {}'.format(np.sqrt(mean_squared_error(RUL_truth, RUL_pred))))
print('\nMAE: {}'.format(mean_absolute_error(RUL_truth, RUL_pred)))

# Scoring criteria
def score_cal(y_hat, Y_test):
    d = np.array(y_hat - Y_test)
    tmp = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        if d[i, 0] >= 0:
            tmp[i] = np.exp(d[i, 0]/10) - 1
        else:
            tmp[i] = np.exp(-d[i, 0]/13) - 1
    return tmp

score_i_tr = score_cal(np.array(RUL_pred).reshape(-1, 1), np.array(RUL_truth).reshape(-1, 1))
print('Score = ' + str(sum(score_i_tr)))

import math
RUL = []
for i in tqdm(range(100)):
    RUL_temp = Simple_RUL_trajectory(id_number = (i+1), top_numbers=7, id_min_distance=min_distance[i], k=0.385)
    RUL.append(RUL_temp)

RUL_pred = [round(i,1) for i in RUL]
print(RUL_pred)

from sklearn.metrics import confusion_matrix, recall_score, precision_score, mean_squared_error, mean_absolute_error

def mape(actual, predict):
    tmp, n = 0.0, 0
    for i in range(0, len(actual)):
        tmp += math.fabs(actual[i]-predict[i])/actual[i]
        n += 1
    return (tmp/n)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print('\nRMSE: {}'.format(np.sqrt(mean_squared_error(RUL_truth, RUL_pred))))
print('\nMAE: {}'.format(mean_absolute_error(RUL_truth, RUL_pred)))
print('\nMAE: {}'.format(mean_absolute_percentage_error(RUL_truth, RUL_pred)))
print('\nMAE: {}'.format(mape(RUL_truth, RUL_pred)))

# Scoring criteria
def score_cal(y_hat, Y_test):
    d = np.array(y_hat - Y_test)
    tmp = np.zeros(d.shape[0])
    for i in range(d.shape[0]):
        if d[i, 0] >= 0:
            tmp[i] = np.exp(d[i, 0]/10) - 1
        else:
            tmp[i] = np.exp(-d[i, 0]/13) - 1
    return tmp

score_i_tr = score_cal(np.array(RUL_pred).reshape(-1, 1), np.array(RUL_truth).reshape(-1, 1))
print('Score = ' + str(sum(score_i_tr)))

# Forecast chart
fig_verify = plt.figure(figsize=(18, 5))
plt.plot(RUL_pred, color='blue')
plt.plot(RUL_truth, color='green')
plt.title('prediction')
plt.ylabel('value')
plt.xlabel('row')
plt.legend(['predicted', 'actual data'], loc='upper left')
plt.show()

# Error histogram
d = np.array(RUL_pred).reshape(-1, 1) - np.array(RUL_truth).reshape(-1, 1)
plt.hist(d, bins=20)
plt.title('Error distribution - Test Set')
plt.ylabel('f')
plt.xlabel('Error: $RUL_{hat}$ - RUL')
plt.show()