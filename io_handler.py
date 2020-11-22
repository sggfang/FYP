import itertools
import csv
import random

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

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


def get_random(origin_index, num):
    times = 3000
    random_list = []
    for i in range(times):
        temp_list = []
        for i in range(num):
            temp = random.randint(1,1000)
            while temp in random_list:
                temp = random.randint(1, 1001)
            temp_list.append(temp)
        random_list.append(temp_list)
    return random_list


if __name__ == '__main__':
    origin_index = []
    index_file = []
    index = 0
    for i in range(1,1001):
        origin_index.append(i)
    for i in get_random(origin_index, 50):
        index_file.append(i)
        index += 1

        AMDs, energy = loaddata(i)
        AMDs_train, AMDs_test, energy_train, energy_test = \
            train_test_split(AMDs, energy, test_size=5179, random_state=2018)
        print(np.array(AMDs_train).shape)
        print(np.array(AMDs_test).shape)

        linreg = LinearRegression()
        linreg.fit(AMDs_train, energy_train)
        #print(linreg.intercept_)
        #print(linreg.coef_)

        energy_pred = linreg.predict(AMDs_test)

        # print("---------------------")
        # print("MSE:", mean_squared_error(energy_test, energy_pred))
        # print("RMSE:", np.sqrt(mean_squared_error(energy_test, energy_pred)))
        # print("R2_score:", r2_score(energy_test, energy_pred))

        fig, ax = plt.subplots()
        ax.scatter(energy_test, energy_pred)
        ax.plot([np.min(energy_test), np.max(energy_test)],
                [np.min(energy_test), np.max(energy_test)],
                'k--', lw=4)
        ax.set_xlabel('Given')
        ax.set_ylabel('Predicted')
        plt.savefig('./image/pvsg_'+str(index)+'.jpg')
        #plt.show()

    with open("data_set.csv", 'w', newline="") as file:
        csv_writer = csv.writer(file)
        for arr in index_file:
            csv_writer.writerow(arr)
