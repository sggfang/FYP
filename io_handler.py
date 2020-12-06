import itertools
import csv
import random

import lr as lr
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

amd_file = 'T2Reduced_AllAtoms_AMD1000.csv'
energy_file = 't2l_energies.csv'
SIZE = 50

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
            temp_amd.append(getattr(row, column_name))
        AMDs.append(temp_amd)
    print("Reading AMDs completed!")

    df = pd.read_csv(energy_file)
    print("Start to read energy......")
    for row in df.itertuples():
        energy.append(getattr(row, 'ENERGY'))
    print("Reading energy completed!")
    return (AMDs, energy)


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
            for i in range(1, 51):
                temp.append(i)
            random_list.append(temp)
    return random_list


def split_cases(AMDs, energy, train_size=4500):
    indexs = []
    train_AMDs = []
    test_AMDs = []
    train_energy = []
    test_energy = []
    for i in range(train_size):
        temp = random.randint(2, 5680)
        while temp in indexs:
            temp = random.randint(2, 5680)
        indexs.append(temp)
    for e in range(len(energy)):
        if (e+2) in indexs:
            train_AMDs.append(AMDs[e])
            train_energy.append(energy[e])
        else:
            test_AMDs.append(AMDs[e])
            test_energy.append(energy[e])
    return indexs, train_AMDs, test_AMDs, \
           train_energy, test_energy


def MSE(target, predictions):
    squared_deviation = np.power(target - predictions, 2)
    return np.mean(squared_deviation)


def MAD(target, predictions):
    absolute_deviation = np.abs(target - predictions)
    return np.mean(absolute_deviation)


if __name__ == '__main__':
    origin_index = []
    index_file = []
    index = 0
    for i in range(1,1001):
        origin_index.append(i)
    for i in get_random(origin_index, 50, random_off=True):
        index += 1
        AMDs, energy = loaddata(i)
        # AMDs_train, AMDs_test, energy_train, energy_test = \
        #     train_test_split(AMDs, energy, test_size=5179)
        AMDs_train, AMDs_test, energy_train, energy_test = \
            train_test_split(AMDs, energy, test_size=0.2, random_state=index)
        # index_file.append(indexs)
        print(np.array(AMDs_train).shape)
        print(np.array(AMDs_test).shape)

        linreg = LinearRegression(normalize=True, n_jobs=-1)
        linreg.fit(AMDs_train, energy_train)
        a, b = linreg.coef_, linreg.intercept_
        #print(linreg.intercept_)
        #print(linreg.coef_)

        energy_pred = linreg.predict(AMDs_test)
        # print("---------------------")
        # print("MSE:", mean_squared_error(energy_test, energy_pred))
        # print("RMSE:", np.sqrt(mean_squared_error(energy_test, energy_pred)))
        # print("R2_score:", r2_score(energy_test, energy_pred))

        fig, ax = plt.subplots()
        ax.scatter(energy_test, energy_pred)
        #plt.plot(range(len(energy_pred)), energy_pred, 'b', label="predict")
        #plt.plot(range(len(energy_train)), energy_train, 'r', label='test')
        ax.plot([np.min(energy_test), np.max(energy_test)],
                [np.min(energy_test), np.max(energy_test)],
                'k--', lw=4)
        ax.set_xlabel('Given')
        ax.set_ylabel('Predicted')
        plt.savefig('./image/pvsg_'+str(index)+'.jpg')
        ##############
        f, ax = plt.subplots(figsize=(7, 5))
        f.tight_layout()
        ax.hist((energy_test - energy_pred)/energy_test,
                bins=40, label='diff', color='b', alpha=.5)
        ax.set_title("Difference between prediction and test values")
        ax.set_xlabel('Diff_percentage')
        ax.set_ylabel('%')
        ax.legend(loc='best')
        plt.savefig('./image_diff/diff_'+str(index)+'.jpg')
        print("MSE:", MSE(energy_test, energy_pred))
        print("MAD:", MAD(energy_test, energy_pred))
        # Draw the line && Display the data sets

        # plt.plot([min(AMDs_train), max(AMDs_train)], [energy_pred[0][0], energy_pred[-1][0]],
        #          label='Prediction Data')
        # # plt.scatter(x_prediction,y_prediction,label = 'Prediction Data')
        # plt.scatter(AMDs_train, energy_train, label='Train Data')
        # plt.scatter(AMDs_test, energy_test, label='Test Data')
        # plt.xlabel("input_parameter_name")
        # plt.ylabel("output_parameter_name")
        # plt.legend()
        # plt.title("Single Value Linear Regression Data")
        # plt.show()

    with open("data_set.csv", 'w', newline="") as file:
        csv_writer = csv.writer(file)
        for arr in index_file:
            csv_writer.writerow(arr)