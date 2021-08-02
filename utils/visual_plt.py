# import API
from keras import backend as K
import numpy as np
import itertools
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
class_list = ['Anterior','Posterior','No-Lacune', 'Lacune','No-Stroke','Stroke']
# AUROC
def AUROC(y_val2, Results2, class_n='Lacune'):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_val2, Results2)
    auc_keras = auc(fpr_keras, tpr_keras)
    plt.figure(figsize=(8,6))
    plt.plot([0.00, 1.00], [0.00, 1.00], 'k--')
    plt.plot(fpr_keras, tpr_keras, label = '{0} Class (area = {1:.3f})'.format(class_n, auc_keras))

    plt.xlabel('False positive rate', fontsize=15)
    plt.ylabel('True positive rate', fontsize=15)
    plt.title('ROC curve', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.show()

# PRROC
# metric PRROC S2

def PRROC(y_val2, Results2, class_n='Lacune'):
    precision, recall, _ = precision_recall_curve(y_val2, Results2)
    # F1-score
    # f1_score = 2*(precision*recall)/(precision+recall+K.epsilon())

    AP_score = average_precision_score(y_val2, Results2)
    plt.figure(figsize=(8,6))
    plt.plot([np.max(recall), np.min(precision)], [np.min(recall), 1.0], 'k--')
    plt.plot(precision, recall,
             label='{0} Class (AP = {1:.5f})'.format(class_n, AP_score))
    plt.xlabel('Recall rate', fontsize=15)
    plt.ylabel('Precision rate', fontsize=15)
    plt.title(f'only S2 stage PR curve', fontsize=15)
    plt.legend(loc='best', fontsize=15)
    plt.show()

# ConfusionMatrix
def plot_confusion_matrix(cm, classes, ths, normalize=True, cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    title=f'Confusion matrix({ths})'
    plt.title(title, fontsize=23)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90,fontsize=15)
    plt.yticks(tick_marks, classes,fontsize=15)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=20)
    
    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label',fontsize=20)
    plt.tight_layout()

def CM(y_vals, Results, ths, class_n):
    target_names = ['','']
    i=0
    if class_n=='Lacune':
        i=2
    elif class_n=='S1':
        i=4
    else:
        pass
    target_names[0]=class_list[0+i]
    target_names[1]=class_list[1+i]
    argmax_pred = np.reshape((Results>ths).astype(int),(len(Results)))
    argmax_truth = y_vals.astype(int)
    cm = confusion_matrix(argmax_truth, argmax_pred)
    np.set_printoptions(precision=2)
    species = [target_names[0], target_names[1]]
    plt.figure(figsize=(6,6))
    plot_confusion_matrix(cm, species, ths)
    print(cm)
    
    
# Threshold
def metric_batchs(y_true_in, y_pred_in, dtype, class_n, axis=0):
    batch_size = y_true_in.shape[0]
    _metric = []
    if dtype !='S1':
        cm = confusion_matrix(y_true_in, y_pred_in)
        np.set_printoptions(precision=2)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        _metric = cm[axis,axis]
    elif dtype =='S1':
        m = tf.keras.metrics.MeanIoU(num_classes=2)
        m.update_state(y_true_in.astype(np.float32),y_pred_in.astype(np.float32))
        _metric.append(m.result().numpy())  
    return _metric

def thersholds(y_val2, Results2, class_n="Lacune"):
    target_names = ['','']
    i=0
    if class_n=='Lacune':
        i=2
    elif class_n=='S1':
        i=4
    else:
        pass
    target_names[0]=class_list[0+i]
    target_names[1]=class_list[1+i]
    
    thresholds_f = np.linspace(0.01, 0.99, 2000)
    thresholds_t = np.linspace(0.01, 0.99, 2000)
    dtype=f'{class_n} acc'
    F_class = np.array([metric_batchs(y_val2, Results2 > threshold, dtype, class_n, 0) for threshold in tqdm(thresholds_f)])
    T_class = np.array([metric_batchs(y_val2, Results2 > threshold, dtype, class_n,1) for threshold in tqdm(thresholds_t)])

    threshold_best_index_f = np.argmax(F_class) 
    iou_best_f = F_class[threshold_best_index_f]
    threshold_best_f = thresholds_f[threshold_best_index_f]

    threshold_best_index_t = np.argmax(T_class) 
    iou_best_t = T_class[threshold_best_index_t]
    threshold_best_t = thresholds_t[threshold_best_index_t]

    idx = np.argwhere(np.diff(np.sign(F_class - T_class))).flatten()

    return thresholds_f[idx]
# ClassificationReport
def ClassReport(Results2, thresholds_f, y_val2, class_n="Lacune"):
    target_names = ['','']
    i=0
    if class_n=='Lacune':
        i=2
    elif class_n=='S1':
        i=4
    else:
        pass
    target_names[0]=class_list[0+i]
    target_names[1]=class_list[1+i]
    y_pred = np.reshape((Results2>thresholds_f).astype(int),(len(Results2)))
    print(classification_report(y_val2, y_pred, target_names=target_names))

# S1+S2-Heatmap




