import os
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from multiprocess import Pool
import logging


def per_class_recall(confusion_matrix):
    """Compute the per class recall for seg&motion.."""
    return np.diag(confusion_matrix) / confusion_matrix.sum(1)


def per_class_precision(confusion_matrix):
    """Compute the per class precision for seg&motion.."""
    return np.diag(confusion_matrix) / confusion_matrix.sum(0)


def process(example):
    file_path, thres = example
    data = np.load(file_path)
    pred = data["pred"]
    gt = data["gt"]
    
    pred_cls = np.zeros(pred.shape[0])
    pred_cls[pred[:, 0] < thres] = 1
    gt[gt > 0] = 1

    cm = confusion_matrix(
        y_true=gt,
        y_pred=pred_cls,
        labels=list(range(3)),
    )

    # print(cm)

    return cm

def mp_process(example_list):
    with Pool(8) as pp:
        results = pp.map(process, example_list)

    reduce_cm = None
    for res in results:
        if reduce_cm is None:
            reduce_cm = res
        else:
            reduce_cm += res
    
    return reduce_cm


def fscore(p, r, beta=1):
    return (1 + beta * beta) * (p * r) / (beta * beta * p  + r)


if __name__ == "__main__":
    # logging.basicConfig(filename='/Users/camilo.tian/workspace/rain_dust/logs/pr.log', filemode='w', level=logging.DEBUG)
    work_dir = "/Users/camilo.tian/workspace/rain_dust/temp/debug_show"
    fn_list = os.listdir(work_dir)
    fp_list = [os.path.join(work_dir, x) for x in fn_list]

    thres_list = np.linspace(0.01, 1.0, num=21)
    class_idx = 1
    all_p = []
    all_r = []
    all_f2 = []
    for thres in thres_list:
        example_list = []
        for fp in fp_list:
            example_list.append([fp, thres])
        reduced_cm = mp_process(example_list)

        precision = per_class_precision(reduced_cm)
        recall = per_class_recall(reduced_cm)
        f1_score = fscore(precision[class_idx], recall[class_idx], 1)
        f2_score = fscore(precision[class_idx], recall[class_idx], 2)

        print("thres: {:.4f} p: {:.2f} r: {:.2f}, f1: {:.2f}, f2: {:.2f}".format(
            thres, precision[class_idx], recall[class_idx], f1_score, f2_score))
        all_p.append(precision[1])
        all_r.append(recall[1])
        all_f2.append(f2_score)
    
    auc = 0
    for i in range(len(thres_list)):
        auc += all_p[i] * all_r[i] * 0.05
    print(auc)

plt.plot(all_r, all_p)
plt.show()

