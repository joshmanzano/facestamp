import pathlib
import pickle

def get_acc(data):
    total = data['tp'] + data['tn'] + data['fp'] + data['fn']
    acc = (data['tp'] + data['tn']) / total
    return acc * 100

def get_fpr(data):
    fpr = data['fp'] / (data['fp'] + data['tn'])
    return fpr * 100

def get_fnr(data):
    fnr = data['fn'] / (data['fn'] + data['tp'])
    return fnr * 100

if __name__ == '__main__':
    name = 'testing_results_1640784650.bin'
    # name = 'testing_results_1640780450.bin'
    results = pickle.load(open(name,'rb'))
    strings = []
    for result in results:
        print('###',result,'###')
        data = results[result]
        acc = get_acc(data['acc'])
        fpr = get_fpr(data['acc'])
        fnr = get_fnr(data['acc'])
        tp = data['acc']['tp']
        tn = data['acc']['tn']
        fn = data['acc']['fn']
        fp = data['acc']['fp']
        preswap_mean = data['preswap_match'][0] * 100
        preswap_std = data['preswap_match'][1] * 100
        swap_mean = data['swap_match'][0] * 100
        swap_std = data['swap_match'][1] * 100
        print(data)
        strings.append([result, preswap_mean, preswap_std, swap_mean, swap_std, acc, fpr, fnr, tp, tn, fn, fp])
    with open('output_result.csv', 'w') as f:
        for string in strings:
            string = [str(i) for i in string]
            f.write(','.join(string) + '\n')