from termcolor import colored
from tensorflow import keras
import tensorflow as tf, numpy as np, cv2
from keras import backend as K
import matplotlib.pyplot as plt
from PIL import Image as im
from sklearn.metrics import confusion_matrix
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm
class_list = ['Anterior','Posterior','Lacune', 'Non-Lacune']
def metric_batchs(y_true_in, y_pred_in, dtype, class_n, axis=0):
    batch_size = y_true_in.shape[0]
    _metric = []
    if dtype ==f'{class_n} acc':
        cm = confusion_matrix(y_true_in, y_pred_in)
        np.set_printoptions(precision=2)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        _metric = cm[axis,axis]
    return _metric

def thersholds(y_val2, Results2, class_n="Lacune"):
    target_names = ['','']
    i=0
    if class_n=='Lacune':
        i=2
    else:
        pass
    target_names[0]=class_list[0+i]
    target_names[1]=class_list[1+i]
    
    thresholds_f = np.linspace(0.01, 0.99, 2000)
    thresholds_t = np.linspace(0.01, 0.99, 2000)
    dtype=f'{class_n} acc'
    F_class = np.array([metric_batchs(y_val2, Results2 > threshold, dtype, class_n, 0) for threshold in tqdm(thresholds_f)])
    T_class = np.array([metric_batchs(y_val2, Results2 > threshold, dtype, class_n,1) for threshold in tqdm(thresholds_t)])
    idx = np.argwhere(np.diff(np.sign(F_class - T_class))).flatten()
    print(f'Threshold of Best Intersection: ({thresholds_f[idx]} mean acc > {F_class[idx]}) ')
    return thresholds_f[idx]


def heat_map(model2, img_data, truth_label, mri_name , threshold, class_n = 'Lacune', last_conv_n = 'conv3d_19',):
    img_data = np.expand_dims(img_data, axis=0)
#     print(img_data.shape)
    img_data = np.expand_dims(img_data, axis=-1)
    img_data = np.expand_dims(img_data, axis=-1)
#     print(img_data.shape)
    data = np.transpose(img_data)
#     print(img_data.shape)
    preds = model2.predict(img_data)
    pred_class = (preds>threshold).astype(int)
    
#     -------------------------------------------------
    if class_n=='Lacune':
        pre_class_list2 = ['No-Lacune','Lacune']
    else:
        pre_class_list2 = ['Anterior','Posterior']
#     -------------------------------------------------

    data_type = [pre_class_list2]
    pre_class_list = data_type[0]

    lacune_class = model2.output
    # last conv layer
    conv_layer = model2.get_layer(last_conv_n)
    heatmap_model = keras.Model([model2.inputs],[conv_layer.output,model2.output])

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = heatmap_model(img_data)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2, 3))
    # print(pooled_grads.shape)
    last_conv_layer_output = last_conv_layer_output[0]
    # print(last_conv_layer_output.shape)
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    # hp_L = cv2.resize(heatmap, (384,384), interpolation=cv2.INTER_CUBIC)
    # hp_L = im.fromarray(hp_L)
    # hp_L = np.array(hp_L)
    return heatmap