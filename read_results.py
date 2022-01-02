import pathlib
import pickle
import numpy as np
# from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import metrics

def get_acc(data):
    total = data['tp'] + data['tn'] + data['fp'] + data['fn']
    acc = (data['tp'] + data['tn']) / total
    return acc

def get_fpr(data):
    fpr = data['fp'] / (data['fp'] + data['tn'])
    return fpr

def get_fnr(data):
    fnr = data['fn'] / (data['fn'] + data['tp'])
    return fnr

def get_tpr(data):
    tpr = data['tp'] / (data['tp'] + data['fn'])
    return tpr


if __name__ == '__main__':
    name = 'testing_results.bin'
    # name = 'testing_results_1640780450.bin'
    results = pickle.load(open(name,'rb'))
    # strings = []
    for result in results:
        preswap = results[result]['preswap_match']
        swap = results[result]['swap_match']
        preswap_true = [1 for _ in range(len(preswap))]
        swap_true = [0 for _ in range(len(swap))]
        gt = preswap_true + swap_true
        prob = preswap + swap
        auc = metrics.roc_auc_score(gt, prob)
        fpr, tpr, _ = metrics.roc_curve(gt, prob)
        plt.plot(fpr, tpr, label=f'{result}')
        print(f'{result}: {auc}')
    #     data = results[result]
    #     acc_data = data['acc']
    #     tprs = []
    #     fprs = []
    #     for threshold in acc_data:
    #         acc = get_acc(acc_data[threshold])
    #         tpr = get_tpr(acc_data[threshold]) 
    #         fpr = get_fpr(acc_data[threshold])
    #         fnr = get_fnr(acc_data[threshold])
    #         tprs.append(tpr)
    #         fprs.append(fpr)
    #             # auc_data[threshold][point] = acc[point][threshold]
            
    #     print('##' + result + '##')
    #     preswap_mean = data['preswap_match'][0] * 100
    #     preswap_std = data['preswap_match'][1] * 100
    #     swap_mean = data['swap_match'][0] * 100
    #     swap_std = data['swap_match'][1] * 100
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(f'Receiver Operating Characteristic Graph')
    plt.legend()
    plt.savefig(f'results/results_auc.png')
    plt.close()

    fig = plt.figure()
    for result in results:
        preswap = results[result]['preswap_match']
        swap = results[result]['swap_match']
        preswap_true = [1 for _ in range(len(preswap))]
        swap_true = [0 for _ in range(len(swap))]
        gt = preswap_true + swap_true
        prob = preswap + swap
        auc = metrics.roc_auc_score(gt, prob)
        fpr, tpr, _ = metrics.roc_curve(gt, prob)
        plt.plot(fpr, tpr, label=f'{result}')
        print(f'{result}: {auc}')
    #     data = results[result]
    #     acc_data = data['acc']
    #     tprs = []
    #     fprs = []
    #     for threshold in acc_data:
    #         acc = get_acc(acc_data[threshold])
    #         tpr = get_tpr(acc_data[threshold]) 
    #         fpr = get_fpr(acc_data[threshold])
    #         fnr = get_fnr(acc_data[threshold])
    #         tprs.append(tpr)
    #         fprs.append(fpr)
    #             # auc_data[threshold][point] = acc[point][threshold]
            
    #     print('##' + result + '##')
    #     preswap_mean = data['preswap_match'][0] * 100
    #     preswap_std = data['preswap_match'][1] * 100
    #     swap_mean = data['swap_match'][0] * 100
    #     swap_std = data['swap_match'][1] * 100
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title(f'Receiver Operating Characteristic Graph')
    plt.legend()
    plt.savefig(f'results/results_auc.png')
    plt.close()
        
        # print(auc_data)

        # tp = data['acc']['tp']
        # tn = data['acc']['tn']
        # fn = data['acc']['fn']
        # fp = data['acc']['fp']


        # print(data)
        # strings.append([result, preswap_mean, preswap_std, swap_mean, swap_std, acc, fpr, fnr, tp, tn, fn, fp])
    # with open('output_result.csv', 'w') as f:
    #     for string in strings:
    #         string = [str(i) for i in string]
    #         f.write(','.join(string) + '\n')