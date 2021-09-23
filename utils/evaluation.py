from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, confusion_matrix
import numpy as np
from scipy.stats import rankdata


def iou_score(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / (np.sum(union) + 1e-6)
    return iou_score

def dice_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    im_sum = actual.sum() + predicted.sum()
    if im_sum == 0: return 1
    intersection = np.logical_and(actual, predicted)
    return 2. * intersection.sum() / im_sum

def accuracy_score(actual, predicted):
    actual = np.asarray(actual).astype(np.bool)
    predicted = np.asarray(predicted).astype(np.bool)
    num_els = actual.size
    intersection = np.logical_and(actual, predicted)
    return float(intersection.sum()) / num_els

def fast_auc(actual, predicted):
    r = rankdata(predicted)
    n_pos = np.sum(actual)
    n_neg = len(actual) - n_pos
    return (np.sum(r[actual==1]) - n_pos*(n_pos+1)/2) / (n_pos*n_neg)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None, text_file=None):
    """
    pretty print for confusion matrixes
    https://gist.github.com/zachguo/10296432
    """
    if text_file is None: print("\n", end=" ")
    else: print("\n", end=" ", file=open(text_file, "a"))

    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    fst_empty_cell = (columnwidth - 3) // 2 * " " + "t/p" + (columnwidth - 3) // 2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    if text_file is None: print("    " + fst_empty_cell, end=" ")
    else: print("    " + fst_empty_cell, end=" ", file = open(text_file, "a"))

    for label in labels:
        if text_file is None: print("%{0}s".format(columnwidth) % label, end=" ")
        else: print("%{0}s".format(columnwidth) % label, end=" ", file = open(text_file, "a"))
    if text_file is None: print()
    else: print(' ', file = open(text_file, "a"))
    # Print rows
    for i, label1 in enumerate(labels):
        if text_file is None: print("    %{0}s".format(columnwidth) % label1, end=" ")
        else: print("    %{0}s".format(columnwidth) % label1, end=" ", file = open(text_file, "a"))
        for j in range(len(labels)):
            cell = "%{}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            if text_file is None: print(cell, end=" ")
            else: print(cell, end=" ", file = open(text_file, "a"))
        if text_file is None: print()
        else: print(' ', file = open(text_file, "a"))

def evaluate_multi_cls(y_true, y_pred, y_proba, print_conf=True, text_file=None, class_names=None):
    classes, _ = np.unique(y_true, return_counts=True)
    if class_names is None:
        class_names = [str(n) for n in classes]

    f1 = f1_score(y_true, y_pred, average='micro')
    mcc = matthews_corrcoef(y_true, y_pred)
    if len(classes)==2:
        mean_auc = roc_auc_score(y_true, y_proba[:,1])
    else:
        mean_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovo')

    # mean_auc = roc_auc_score(y_true, y_proba, average='weighted', multi_class='ovr')
    # ovo should be better, but average is not clear from docs
    # mean_auc = roc_auc_score(y_true, y_proba, average='macro', multi_class='ovo')

    if print_conf:
        if text_file is not None:
            print("\nMCC={:.2f} -- F1={:.2f} -- AUC={:.2f}".format(100*mcc, 100*f1, 100*mean_auc), end=" ", file=open(text_file, "a"))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        print_cm(cm, class_names, text_file=text_file)

    return mean_auc, mcc, f1