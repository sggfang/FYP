# !/usr/bin/env python
import argparse
import math
import random

import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn import neighbors
from sklearn import tree
from sklearn.cluster import AffinityPropagation, AgglomerativeClustering, Birch, MiniBatchKMeans, \
    MeanShift
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from sklearn.neighbors._ball_tree import BallTree
from sklearn.preprocessing import PolynomialFeatures

amd_file = 'T2Reduced_AllAtoms_AMD1000.csv'
energy_file = 't2l_energies.csv'
SIZE = 50
TEST_NUM = 1136
ATTRIBUTE_NUM = 50


def parse_augument():
    parser = argparse.ArgumentParser()
    parser.add_argument('type', help='designate the model type')
    return parser


def loaddata(nums):
    names = []
    AMDs = []
    energy = []
    df = pd.read_csv(amd_file)
    print("Start to read AMDs......")
    for row in df.itertuples():
        crystal_name = getattr(row, "ID")
        names.append(crystal_name)
        temp_amd = []
        for i in nums:
            column_name = "AMD_" + str(i)
            temp_amd.append(100 * getattr(row, column_name))
        AMDs.append(temp_amd)
    print("Reading AMDs completed!")

    df = pd.read_csv(energy_file)
    print("Start to read energy......")
    for row in df.itertuples():
        energy.append(getattr(row, 'ENERGY'))
    print("Reading energy completed!")
    return AMDs, energy


def get_random(origin_index, num, random_off=False):
    random_list = []
    times = 1
    if not random_off:
        for j in range(times):
            temp_list = []
            for i in range(num):
                temp = random.randint(1, 1000)
                while temp in random_list:
                    temp = random.randint(1, 1001)
                temp_list.append(temp)
            random_list.append(temp_list)
    else:
        for j in range(times):
            temp = []
            for i in range(1, 1 + num):
                temp.append(i)
            random_list.append(temp)
    return random_list


def to_number(y, position):
    return str(int(1136 * y))


def pack_data(AMDs_train, AMDs_test,
              energy_train, energy_test):
    train_packed = []
    test_packed = []
    AMDs_train_packed = []
    AMds_test_packed = []
    index = 0
    for arr in AMDs_train:
        sum = 0
        for e in arr:
            sum += e * e
        train_packed.append([sum, energy_train[index]])
        AMDs_train_packed.append(sum)
        index += 1
    index = 0
    for arr in AMDs_test:
        sum = 0
        for e in arr:
            sum += e * e
        test_packed.append([sum, energy_test[index]])
        AMds_test_packed.append(sum)
        index += 1

    return train_packed, test_packed, \
           AMDs_train_packed, AMds_test_packed


def LINEAR(AMDs_train_packed, AMDs_test_packed, AMDs_train,
           AMDs_test, energy_train, energy_test):
    clusters_amd_1 = []
    clusters_amd_2 = []
    clusters_amd_3 = []
    clusters_amd_4 = []
    clusters_energy_1 = []
    clusters_energy_2 = []
    clusters_energy_3 = []
    clusters_energy_4 = []
    i = 0
    for e in AMDs_train_packed:
        if e <= 280:
            clusters_amd_1.append(AMDs_train[i])
            clusters_energy_1.append(energy_train[i])
        elif 280 < e <= 300:
            clusters_amd_2.append(AMDs_train[i])
            clusters_energy_2.append(energy_train[i])
        elif 300 < e <= 320:
            clusters_amd_3.append(AMDs_train[i])
            clusters_energy_3.append(energy_train[i])
        elif 320 < e:
            clusters_amd_4.append(AMDs_train[i])
            clusters_energy_4.append(energy_train[i])
        i += 1

    clusters_amd_test_1 = []
    clusters_amd_test_2 = []
    clusters_amd_test_3 = []
    clusters_amd_test_4 = []
    clusters_energy_test_1 = []
    clusters_energy_test_2 = []
    clusters_energy_test_3 = []
    clusters_energy_test_4 = []
    temp_energy_test = []
    i = 0
    for e in AMDs_test_packed:
        if e <= 280:
            clusters_amd_test_1.append(AMDs_test[i])
            clusters_energy_test_1.append(energy_test[i])
        elif 280 < e <= 300:
            clusters_amd_test_2.append(AMDs_test[i])
            clusters_energy_test_2.append(energy_test[i])
        elif 300 < e <= 320:
            clusters_amd_test_3.append(AMDs_test[i])
            clusters_energy_test_3.append(energy_test[i])
        elif 320 < e:
            clusters_amd_test_4.append(AMDs_test[i])
            clusters_energy_test_4.append(energy_test[i])
        i += 1
    # Linear regression
    linreg_1 = LinearRegression(normalize=True, n_jobs=-1)
    linreg_1.fit(clusters_amd_1, clusters_energy_1)
    energy_pred_1 = linreg_1.predict(clusters_amd_test_1)
    linreg_2 = LinearRegression(normalize=True, n_jobs=-1)
    linreg_2.fit(clusters_amd_2, clusters_energy_2)
    energy_pred_2 = linreg_2.predict(clusters_amd_test_2)
    linreg_3 = LinearRegression(normalize=True, n_jobs=-1)
    linreg_3.fit(clusters_amd_3, clusters_energy_3)
    energy_pred_3 = linreg_3.predict(clusters_amd_test_3)
    linreg_4 = LinearRegression(normalize=True, n_jobs=-1)
    linreg_4.fit(clusters_amd_4, clusters_energy_4)
    energy_pred_4 = linreg_4.predict(clusters_amd_test_4)

    energy_pred = []
    energy_pred.extend(energy_pred_1)
    energy_pred.extend(energy_pred_2)
    energy_pred.extend(energy_pred_3)
    energy_pred.extend(energy_pred_4)
    temp_energy_test = []
    temp_energy_test.extend(clusters_energy_test_1)
    temp_energy_test.extend(clusters_energy_test_2)
    temp_energy_test.extend(clusters_energy_test_3)
    temp_energy_test.extend(clusters_energy_test_4)

    fig, ax = plt.subplots()
    print("MSR of clustering is: ",
          mean_squared_error(temp_energy_test, energy_pred))
    ax.scatter(temp_energy_test, energy_pred)
    ax.plot([np.min(energy_test), np.max(energy_test)],
            [np.min(energy_test), np.max(energy_test)],
            'k--', lw=4)
    ax.set_xlabel('Given')
    ax.set_ylabel('Predicted')
    plt.savefig('./image/cluster_lin_' + str(index) + '.jpg')


def nn_linear_regression(AMDs_train, AMDs_test,
                         energy_train, energy_test):
    mse_min = 100
    r2_max = 0
    total_number = ATTRIBUTE_NUM
    weight_list = [0.008, 0.04, 0.2, 1, 5, 25, 125]
    import random
    while True:
        remains = total_number
        center_number = random.randint(0, (int)(remains / 2)) * 2
        remains -= center_number
        number_1 = random.randint(0, (int)(remains / 2)) * 2
        remains -= number_1
        number_2 = random.randint(0, (int)(remains / 2)) * 2
        remains -= number_2
        number_3 = remains

        center_list = []
        list_1 = []
        list_2 = []
        list_3 = []
        for i in range(center_number):
            temp = random.randint(0, total_number - 1)
            while temp in center_list:
                temp = random.randint(0, total_number - 1)
            center_list.append(temp)

        list_1_1 = []
        list_1_2 = []
        for i in range(number_1):
            temp = random.randint(0, total_number - 1)
            while (temp in center_list) \
                    or (temp in list_1):
                temp = random.randint(0, total_number - 1)
            list_1.append(temp)
            if i % 2 == 0:
                list_1_1.append(temp)
            else:
                list_1_2.append(temp)

        list_2_1 = []
        list_2_2 = []
        for i in range(number_2):
            temp = random.randint(0, total_number - 1)
            while (temp in center_list) \
                    or (temp in list_1) \
                    or (temp in list_2):
                temp = random.randint(0, total_number - 1)
            list_2.append(temp)
            if i % 2 == 0:
                list_2_1.append(temp)
            else:
                list_2_2.append(temp)

        for i in range(total_number):
            if i not in center_list \
                    and i not in list_1 \
                    and i not in list_2:
                list_3.append(i)
        list_3_1 = list_3[0: (int)(number_3 / 2)]
        list_3_2 = list_3[(int)(number_3 / 2): number_3]

        new_AMDs_train = []
        for row in AMDs_train:
            temp_row = []
            for i in range(total_number):
                if i in center_list:
                    temp_row.append(weight_list[3] * row[i])
                elif i in list_1_2:
                    temp_row.append(weight_list[4] * row[i])
                elif i in list_1_1:
                    temp_row.append(weight_list[2] * row[i])
                elif i in list_2_2:
                    temp_row.append(weight_list[5] * row[i])
                elif i in list_2_1:
                    temp_row.append(weight_list[1] * row[i])
                elif i in list_3_2:
                    temp_row.append(weight_list[6] * row[i])
                elif i in list_3_1:
                    temp_row.append(weight_list[0] * row[i])
                else:
                    print("index not found: ", i)
            new_AMDs_train.append(temp_row)

        new_AMDs_test = []
        for row in AMDs_test:
            temp_row = []
            for i in range(total_number):
                if i in center_list:
                    temp_row.append(weight_list[3] * row[i])
                elif i in list_1_2:
                    temp_row.append(weight_list[4] * row[i])
                elif i in list_1_1:
                    temp_row.append(weight_list[2] * row[i])
                elif i in list_2_2:
                    temp_row.append(weight_list[5] * row[i])
                elif i in list_2_1:
                    temp_row.append(weight_list[1] * row[i])
                elif i in list_3_2:
                    temp_row.append(weight_list[6] * row[i])
                elif i in list_3_1:
                    temp_row.append(weight_list[0] * row[i])
                else:
                    print("index not found: ", i)
            new_AMDs_test.append(temp_row)

        # Linear regression
        linreg = LinearRegression(normalize=True, n_jobs=-1)
        linreg.fit(new_AMDs_train, energy_train)
        energy_pred = linreg.predict(new_AMDs_test)
        mse = round(mean_squared_error(energy_test, energy_pred), 4)
        r2 = round(r2_score(energy_test, energy_pred), 4)

        print("MSR of weighted linear regression is: ",
              mse)
        print("r2 is ", r2)

        fig, ax = plt.subplots()
        ax.scatter(energy_test, energy_pred)
        ax.plot([np.min(energy_test), np.max(energy_test)],
                [np.min(energy_test), np.max(energy_test)],
                'k--', lw=4)
        ax.set_xlabel('Given')
        ax.set_ylabel('Predicted')
        plt.savefig('./image/wlin_' + str(index) + '.jpg')
        break


def predict(temp_AMDs, temp_AMDs_test, temp_energy):
    mid = np.median(temp_energy)
    linreg = LinearRegression(normalize=True, n_jobs=-1)
    linreg.fit(temp_AMDs, temp_energy)
    temp_energy_pred = linreg.predict(temp_AMDs_test)
    new_energy_pred = []
    for ele in temp_energy_pred:
        if ele > mid:
            diff = ele - mid
            new_energy_pred.append(ele - 0.1 * diff)
        else:
            diff = mid - ele
            new_energy_pred.append(ele + 0.1 * diff)
    return new_energy_pred


def shift(ele, energy_list):
    # min_value = np.min(energy_list)
    # max_value = np.max(energy_list)
    # return (ele - min_value)/(max_value - min_value)
    mean_value = np.mean(energy_list)
    std_value = np.std(energy_list, ddof=1)
    return (ele - mean_value) / std_value


def neighbor_cluster(AMDs_train, AMDs_test,
                     energy_train, energy_test,
                     type):
    energy_test_pred = []
    N = 15
    if type == "chebyshev":
        tree = BallTree(AMDs_train, metric='chebyshev')
    elif type == "euclidean":
        tree = BallTree(AMDs_train, metric='euclidean')
    elif type == "minkowski":
        tree = BallTree(AMDs_train, metric='minkowski')
    elif type == "manhattan":
        tree = BallTree(AMDs_train, metric='manhattan')
    else:
        return

    dist, inds = tree.query(AMDs_test, k=N)
    for ind in inds:
        sum = 0
        for i in ind:
            sum += energy_train[i]
        ave = sum / N
        energy_test_pred.append(ave)

    fig, ax = plt.subplots()
    print("R^2 score of KNN is: ",
          r2_score(energy_test, energy_test_pred))
    print("RMSE of KNN is: ",
          math.sqrt(mean_squared_error(energy_test, energy_test_pred)))
    ax.scatter(energy_test, energy_test_pred)
    ax.plot([np.min(energy_test), np.max(energy_test)],
            [np.min(energy_test), np.max(energy_test)],
            'k--', lw=4)
    ax.set_xlabel('Given')
    ax.set_ylabel('Predicted')
    plt.savefig('./image/knn_' + type + '.jpg')


def combination_algorithm(AMDs_train, energy_train, AMDs_test,
                          energy_test, type):
    NUMBER_OF_CLUSTER = 5
    if type == "kmeans_com":
        model = KMeans(n_clusters=NUMBER_OF_CLUSTER).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "affinity_com":
        model = AffinityPropagation(damping=0.9, random_state=5).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "agglomerative_com":
        model = AgglomerativeClustering(n_clusters=NUMBER_OF_CLUSTER)
        y_clusters = model.fit_predict(AMDs_test)
    elif type == "birch_com":
        model = Birch(threshold=0.1, n_clusters=NUMBER_OF_CLUSTER).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "minibatch_com":
        model = MiniBatchKMeans(n_clusters=NUMBER_OF_CLUSTER).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "meanshift_com":
        model = MeanShift().fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    else:
        return

    new_energy = []
    new_energy_test = []
    for i in range(NUMBER_OF_CLUSTER):
        if i not in y_clusters:
            print("ERROR: ", i, " is not here")
            continue
        index = 0
        temp_AMDs = []
        temp_energy = []
        for j in model.labels_:
            if i == j:
                temp_AMDs.append(AMDs_train[index])
                temp_energy.append(energy_train[index])
            index += 1

        index = 0
        temp_AMDs_test = []
        temp_energy_test = []
        for j in y_clusters:
            if i == j:
                temp_AMDs_test.append(AMDs_test[index])
                temp_energy_test.append(energy_test[index])
            index += 1

        quadratic_featurizer = PolynomialFeatures(degree=1, interaction_only=True)
        X_train_quadratic = quadratic_featurizer.fit_transform(temp_AMDs)
        X_test_quadratic = quadratic_featurizer.fit_transform(temp_AMDs_test)
        model2 = LinearRegression()
        model2.fit(X_train_quadratic, temp_energy)

        temp_energy_pred = model2.predict(X_test_quadratic)

        new_energy.extend(temp_energy_pred)
        new_energy_test.extend(temp_energy_test)

        fig, ax = plt.subplots()
        ax.scatter(temp_energy_test, temp_energy_pred)
        ax.plot([np.min(temp_energy_test), np.max(temp_energy_test)],
                [np.min(temp_energy_test), np.max(temp_energy_test)],
                'k--', lw=4)
        ax.set_xlabel('Given')
        ax.set_ylabel('Predicted')
        plt.savefig('./image/combination_algorithm' + str(i) + '.jpg')

    fig, ax = plt.subplots()
    print("R^2 score of the combination algorithm is: ",
          r2_score(new_energy_test, new_energy))
    print("RMSE of the combination algorithm is: ",
          math.sqrt(mean_squared_error(new_energy_test, new_energy)))
    ax.scatter(new_energy_test, new_energy)
    ax.plot([np.min(new_energy_test), np.max(new_energy_test)],
            [np.min(new_energy_test), np.max(new_energy_test)],
            'k--', lw=4)
    ax.set_xlabel('Given')
    ax.set_ylabel('Predicted')
    plt.savefig('./image/combination_algorithm.jpg')


def regression(AMDs_train, energy_train, AMDs_test,
               energy_test, type):
    if type == "knn_uniform":
        n_neighbors = 15
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights="uniform")
        energy_pred = knn.fit(AMDs_train, energy_train).predict(AMDs_test)
    elif type == "knn_distance":
        n_neighbors = 15
        knn = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        energy_pred = knn.fit(AMDs_train, energy_train).predict(AMDs_test)
    elif type == "linear":
        linreg = LinearRegression(normalize=True, n_jobs=-1)
        linreg.fit(AMDs_train, energy_train)
        energy_pred = linreg.predict(AMDs_test)
    elif type == "poly":
        featurizer = PolynomialFeatures(degree=2)
        X_train = featurizer.fit_transform(AMDs_train)
        X_test = featurizer.fit_transform(AMDs_test)
        model2 = LinearRegression()
        model2.fit(X_train, energy_train)
        energy_pred = model2.predict(X_test)
    else:
        return

    fig, ax = plt.subplots()
    print("RMSE of ", type, " regression is: ",
          math.sqrt(mean_squared_error(energy_test, energy_pred)))
    print("r2 of ", type, " regression is: ",
          r2_score(energy_test, energy_pred))
    ax.scatter(energy_test, energy_pred)
    ax.plot([np.min(energy_test), np.max(energy_test)],
            [np.min(energy_test), np.max(energy_test)],
            'k--', lw=4)
    ax.set_xlabel('Given')
    ax.set_ylabel('Predicted')
    plt.savefig('./image/' + type + '.jpg')


def pure_clustering(AMDs_train, energy_train, AMDs_test,
                    energy_test, type):
    NUMBER_OF_CLUSTER = 100
    if type == "kmeans":
        model = KMeans(n_clusters=NUMBER_OF_CLUSTER).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "affinity":
        model = AffinityPropagation(damping=0.9, random_state=5).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "agglomerative":
        model = AgglomerativeClustering(n_clusters=NUMBER_OF_CLUSTER)
        y_clusters = model.fit_predict(AMDs_test)
    elif type == "birch":
        model = Birch(threshold=0.1, n_clusters=NUMBER_OF_CLUSTER).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "minibatch":
        model = MiniBatchKMeans(n_clusters=NUMBER_OF_CLUSTER).fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    elif type == "meanshift":
        model = MeanShift().fit(AMDs_train)
        y_clusters = model.predict(AMDs_test)
    else:
        return

    new_energy = []
    new_energy_test = []
    for i in range(NUMBER_OF_CLUSTER):
        if i not in y_clusters:
            print("ERROR: ", i, " is not here")
        index = 0
        temp_energy = []
        for j in model.labels_:
            if i == j:
                temp_energy.append(energy_train[index])
            index += 1
        new_energy.append(np.average(temp_energy))

    new_energy_pred = []
    for j in y_clusters:
        new_energy_pred.append(new_energy[j])

    fig, ax = plt.subplots()
    print("R^2 score of pure clustering is: ",
          r2_score(energy_test, new_energy_pred))
    print("RMSE of pure clustering is: ",
          math.sqrt(mean_squared_error(energy_test, new_energy_pred)))
    ax.scatter(energy_test, new_energy_pred)
    ax.plot([np.min(energy_test), np.max(energy_test)],
            [np.min(energy_test), np.max(energy_test)],
            'k--', lw=4)
    ax.set_xlabel('Given')
    ax.set_ylabel('Predicted')
    plt.savefig('./image/cluster_' + type + '.jpg')


if __name__ == '__main__':
    origin_index = []
    index_file = []
    index = 0
    LIN = True
    for i in range(1, 1001):
        origin_index.append(i)

    parser = parse_augument()
    args = parser.parse_args()

    # Run several times to get the average result
    for i in get_random(origin_index, 55, random_off=True):
        index += 1
        AMDs, energy = loaddata(i)
        AMDS_with_header = pd.read_csv(amd_file)
        energy_with_header = pd.read_csv(energy_file)

        AMDs_train, AMDs_test, energy_train, energy_test = \
            train_test_split(AMDs, energy, test_size=0.2, random_state=index)
        AMDs_train_, AMDs_test_, energy_train_, energy_test_ = \
            train_test_split(AMDS_with_header, energy_with_header,
                             test_size=0.2, random_state=index)
        train_packed, test_packed, AMDs_train_packed, AMDs_test_packed = \
            pack_data(AMDs_train, AMDs_test, energy_train, energy_test)

        combination_algorithm(AMDs_train, energy_train, AMDs_test,
                              energy_test, args.type)

        pure_clustering(AMDs_train, energy_train, AMDs_test,
                              energy_test, args.type)

        neighbor_cluster(AMDs_train, AMDs_test,
                         energy_train, energy_test, args.type)

        # LINEAR(AMDs_train_packed, AMDs_test_packed, AMDs_train,
        #        AMDs_test, energy_train, energy_test)

        # nn_linear_regression(AMDs_train, AMDs_test,
        #                      energy_train, energy_test)

        regression(AMDs_train, energy_train, AMDs_test,
                   energy_test, args.type)

        if args.type == "xgboost":
            energy_test_ = energy_test_.drop(['ID'], axis=1)
            energy_train_ = energy_train_.drop(['ID'], axis=1)
            AMDs_test_ = AMDs_test_.drop(['ID'], axis=1)
            AMDs_train_ = AMDs_train_.drop(['ID'], axis=1)
            from sklearn.model_selection import GridSearchCV

            cv_params = {'n_estimators': [120, 150, 200, 300, 600,
                                          700, 900, 1000, 1200, 1500,
                                          2000, 3000, 5000, 8000, 10000],
                         "learning_rate": [0.01],
                         'max_depth': [4, 6, 9, 12, 15]}
            other_params = {'min_child_weight': 1, 'seed': 0, 'subsample': 0.8}
            model = xgb.XGBRegressor(**other_params)
            optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params,
                                         scoring='r2', cv=5, verbose=1, n_jobs=4)
            optimized_GBM.fit(AMDs_train_, energy_train_)
            evalute_result = optimized_GBM.cv_results_
            ans = optimized_GBM.predict(AMDs_test_)

            print("MSE of XGB regression: ",
                  mean_squared_error(energy_test, ans))
            print("r2 of XGB regression is: ",
                  r2_score(energy_test, ans))
            print('Running result after each iteration:{0}'.format(evalute_result))
            print('optimal parameter：{0}'.format(optimized_GBM.best_params_))
            fig, ax = plt.subplots()
            ax.scatter(energy_test_, ans)
            ax.plot([np.min(energy_test), np.max(energy_test)],
                    [np.min(energy_test), np.max(energy_test)],
                    'k--', lw=4)
            ax.set_xlabel('Given')
            ax.set_ylabel('Predicted')
            plt.savefig('./image/xgboost_' + str(index) + '.jpg')

