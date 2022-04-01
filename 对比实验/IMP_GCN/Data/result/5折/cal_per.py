# -*-encoding:utf -*-
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_curve,auc
import matplotlib.pyplot as plt
from numpy import interp


def draw(fpr_list, tpr_list,  mean_fpr, mean_tpr):
    lw = 1
    plt.plot(fpr_list[0], tpr_list[0], color='orange',
             lw=lw, label='ROC fold 1 (AUC  = 0.8968)', linestyle='--')
    plt.plot(fpr_list[1], tpr_list[1], color='green',
             lw=lw, label='ROC fold 2 (AUC = 0.8881)', linestyle='--')
    plt.plot(fpr_list[2], tpr_list[2], color='blue',
             lw=lw, label='ROC fold 3 (AUC = 0.8835)', linestyle='--')
    plt.plot(fpr_list[3], tpr_list[3], color='pink',
             lw=lw, label='ROC fold 4 (AUC = 0.8711)', linestyle='--')
    plt.plot(fpr_list[4], tpr_list[4], color='maroon',
             lw=lw, label='ROC fold 5 (AUC  = 0.8931)', linestyle='--')
    plt.plot(mean_fpr, mean_tpr, color='red', label=r'Mean ROC (AUC  = 0.8865)', lw=lw, alpha=.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()


def cal_perform():
    p_list = []
    f_list = []
    acc_list = []
    r_list = []
    auc_list = []
    aupr_list = []
    fpr_list = []
    tpr_list = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for j in range(0, 5):
        pre_path = f"predict_em64_layer6_group3_{j}.npy"
        ac_path = f"true_em64_layer6_group3_{j}.npy"
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
        # for i in range(len(predict_value)):
        #     if predict_value[i] > 0.25:
        #         predict_value[i] = 1
        #     else:
        #         predict_value[i] = 0
        # print(len(predict_value))
        # acc = accuracy_score(Y_test, predict_value.round())
        # print("acc", acc)
        # acc_list.append(acc)
        # p = precision_score(Y_test, predict_value.round())
        # print("precision", p)
        # p_list.append(p)
        # r = recall_score(Y_test, predict_value.round())
        # r_list.append(r)
        # print("recall", r)
        # f1 = f1_score(Y_test, predict_value.round())
        # f_list.append(f1)
        # print("f1_score", f1)
        # print()
        fpr, tpr, threshold = roc_curve(Y_test, predict_value)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    print("the average of AUC", sum(auc_list)/len(auc_list))
    print("the average of AUPR", sum(aupr_list)/len(aupr_list))
    # print("the average of acc", sum(acc_list)/len(acc_list))
    # print("the average of p", sum(p_list)/len(p_list))
    # print("the average of r", sum(r_list)/len(r_list))
    # print("the average of f", sum(f_list)/len(f_list))
    return fpr_list, tpr_list, mean_fpr, mean_tpr


if __name__ == "__main__":
    fpr_list, tpr_list, mean_fpr, mean_tpr = cal_perform()
    np.save("our_fpr.npy", mean_fpr)
    np.save("our_tpr.npy", mean_tpr)
    draw(fpr_list, tpr_list, mean_fpr, mean_tpr)