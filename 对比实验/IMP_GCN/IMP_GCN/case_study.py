# -*-encoding -*-
import numpy as np

disease_list = []


def read():
    path = "case.npy"
    case_result = np.load(path)
    for i in range(0,194):
        disease_list.append(case_result[:, i])
    diseas_dict= {}

    for data in disease_list:
        sno_dict = {}
        for i, sno in enumerate(data):
            sno_dict[i] = sno
        print(sno_dict)
        # print(len(data))


read()