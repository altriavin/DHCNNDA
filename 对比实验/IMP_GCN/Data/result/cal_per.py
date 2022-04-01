# -*-encoding:utf -*-
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_curve,auc


def cal_perform():
    p_list = []
    f_list = []
    acc_list = []
    r_list = []
    auc_list = []
    aupr_list = []
    embed_list = [8,16,32,128,256]
    for j in embed_list:
        pre_path = f"predict_em{j}_layer6_group3_1.npy"
        ac_path = f"true_em{j}_layer6_group3_1.npy"
        ac = np.load(ac_path)
        # print(ac)
        pre = np.load(pre_path)
        # print(pre)
        Y_test = ac
        predict_value = pre
        AUC = roc_auc_score(ac, pre)
        print("AUC", AUC)
        auc_list.append(AUC)
        precision, recall, _ = precision_recall_curve(Y_test, predict_value)
        # precision, recall, _ = precision_recall_curve(Y_test, predict_value)
        AUCPR = auc(recall, precision)
        print("AUPR", AUCPR)
        aupr_list.append(AUCPR)
        # fpr, tpr, threshold = roc_curve(Y_test, predict_value)
        # np.save(f"data/fpr{j}.npy", fpr)
        # np.save(f"data/tpr{j}.npy", tpr)
        for i in range(len(predict_value)):
            if predict_value[i] > 0.25:
                predict_value[i] = 1
            else:
                predict_value[i] = 0
        # print(len(predict_value))
        acc = accuracy_score(Y_test, predict_value.round())
        print("acc", acc)
        acc_list.append(acc)
        p = precision_score(Y_test, predict_value.round())
        print("precision", p)
        p_list.append(p)
        r = recall_score(Y_test, predict_value.round())
        r_list.append(r)
        print("recall", r)
        f1 = f1_score(Y_test, predict_value.round())
        f_list.append(f1)
        print("f1_score", f1)
        print()
    print("the average of AUC", sum(auc_list)/len(auc_list))
    print("the average of AUPR", sum(aupr_list)/len(aupr_list))
    print("the average of acc", sum(acc_list)/len(acc_list))
    print("the average of p", sum(p_list)/len(p_list))
    print("the average of r", sum(r_list)/len(r_list))
    print("the average of f", sum(f_list)/len(f_list))


if __name__ == "__main__":
    cal_perform()