# only using 3.0T data to train.
import zipfile, os, numpy as np, pickle, yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.ImageDataGenerator import AP_generator
# Display
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, LearningRateScheduler
import tensorflow as tf
tf.executing_eagerly()
from model.resnet3d import Resnet3DBuilder
from tensorflow import keras
from tensorflow.keras import layers, backend as K
from utils.loss_metric import binary_focal_loss, weighted_binarycrossentropy
from utils.loss_metric import dyn_weighted_bincrossentropy
class WarmUpLearningRateScheduler(keras.callbacks.Callback):
    def __init__(self, warmup_batches, init_lr, verbose=1):
            super(WarmUpLearningRateScheduler, self).__init__()
            self.warmup_batches = warmup_batches
            self.init_lr = init_lr
            self.verbose = verbose
            self.batch_count = 0
            self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                    'rate to %s.' % (self.batch_count + 1, lr))

class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        filepath = "saved-model-{epoch:02d}-{val_accuracy:.3f}-{val_auc:.3f}.hdf5"

        mck1 = os.path.join(save_path, f"{checkpoint_name}.hdf5")
        #  ↑save best val loss checkpoint
        mck2 = os.path.join(logs_path, filepath)
        #  ↑save each val loss checkpoint
        mck3 = os.path.join(save_path, f"best-valid-auc_{model_n}-{date_name}.hdf5")
        #  ↑save best val auc checkpoint

        callback_list = [
            ModelCheckpoint(mck1, monitor='val_loss', verbose=1, save_best_only=True),
            ModelCheckpoint(mck2, monitor='val_loss', verbose=1, save_best_only=True),
            ModelCheckpoint(mck3, monitor='val_auc', mode='max', verbose=1, save_best_only=True,),
            TensorBoard(log_dir=tlogdir, histogram_freq=1,embeddings_freq=0,embeddings_layer_names=None,),
            LearningRateScheduler(schedule=self._cosine_anneal_schedule),
            # LearningRateScheduler(self.schedule),
            EarlyStopping(monitor = "val_loss", patience = 10)
            # WarmUpLearningRateScheduler(warmup_batches = 10, init_lr=lr)
        ]
        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

# def model_loader(weight_zero, weight_one):
def model_loader():
    
    # wce = weighted_binarycrossentropy(weight_zero = weight_zero, weight_one = weight_one)
    focal_loss = binary_focal_loss(gamma = config['focal_gamma'], alpha = config['focal_alpha'])
    model = Resnet3DBuilder.build_resnet_18((32, nii_size, nii_size, 1), 1)
    bce = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer='adam',
                #   loss=dyn_weighted_bincrossentropy,
                  loss=focal_loss,
                  metrics=[tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                           tf.keras.metrics.AUC(),
                           tf.keras.metrics.Recall()
                          ],)
    return model

def data_loder(valid_data='3.0T'):
    # path npy 3.0T, 1.5T
    train_y_30T = np.load(MRI_nii_folder_path + 'T3_0_onehot_train.npy')
    if valid_data=='3.0T':
        # 3.0 T nii data processing
        valid_y_30T = np.load(MRI_nii_folder_path + 'T3_0_onehot_valid.npy')
        X_train = np.load(MRI_nii_folder_path+f'/image_mask_arr_3.0T_384_train.npy')
        X_valid = np.load(MRI_nii_folder_path+f'/image_mask_arr_3.0T_384_valid.npy')
        y_train = np.array(train_y_30T).astype(np.int8)
        y_valid = np.array(valid_y_30T).astype(np.int8)
        path_valid = valid_path_30T
        print(y_valid)

    return path_valid, np.expand_dims(X_train, axis=-1), np.expand_dims(X_valid, axis=-1), np.array(y_train).astype(np.int8), np.array(y_valid).astype(np.int8)

def trainer(valid_data='3.0T'): 
    tf.keras.backend.clear_session()
    #     model build
    model = model_loader()

    #     data loader
    path_valid, X_train, X_valid, y_train, y_valid= data_loder(valid_data=valid_data)
    print(path_valid.shape, X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    print(f'train label 0 - 10 : {y_train[0:10]}')
    #     save valid path
    if not os.path.exists(save_path+ f'/history_log/'):
        os.makedirs(save_path + f'/history_log/')
    np.save(config['save_path'] + f'/history_log/{valid_data}valid_path_{date_name}', path_valid)

    #     generateor
    
    train_generator, valid_generator = AP_generator(X_train, y_train, X_valid, y_valid, batch_size)
    print(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape)
    
    

    
    print('-'*30,f'\ntrain data from {MRI_nii_folder_path}\n','-'*30)
    print('-'*30,f'\nFitting model...\n','-'*30)
    snapshot = SnapshotCallbackBuilder(nb_epochs=epochs, nb_snapshots=1, init_lr=lr)
    
    history = model.fit_generator(train_generator,
                        steps_per_epoch = (len(y_train)*3 // batch_size),
                        validation_data = valid_generator,
                        validation_steps = (len(y_valid) // batch_size),
                        epochs=epochs,shuffle=True,verbose=1,
                        callbacks=snapshot.get_callbacks()
                        )

    with open(save_path + f'/history_log/histroy-fold_{date_name}', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    

if __name__ == '__main__':
    def load_config(config_name):
        with open(config_name) as file:
            config = yaml.safe_load(file)

        return config
    config = load_config("utils/model_config_AP.yaml")
    # Model SETTINGS
    model_n = config['model_name']
    class_n = config['class_name']
    # INITIAL SETTINGS
    nii_size = config['nii_size']
    epochs = config['epochs']
    lr = config['lr']
    batch_size = config['batch_size']
    date_name = config['date_name']
    MRI_nii_folder_path = config['MRI_nii_folder_path']
    save_path =config['save_path']
    # checkpoint_name = f"{model_n}-{nii_size}-NL-epochs_{epochs}-lr_{lr}-batch_{batch_size}-dynBCE-test1_{date_name}"
    checkpoint_name = f"{model_n}-{nii_size}-{class_n}-epochs_{epochs}-lr_{lr}-batch_{batch_size}-FL{config['focal_alpha']}-{date_name}"
    tlogdir = os.path.join(config['save_path'], checkpoint_name)
    # load path-url.npy and load label.npy
    train_path_30T = np.load(MRI_nii_folder_path + 'T3_image_mask_path_train.npy')
    valid_path_30T = np.load(MRI_nii_folder_path + 'T3_image_mask_path_valid.npy')

    # creative weight and tensorboard path folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save checkpoint (best/each)
    logs_path = save_path+ f'logs_{date_name}_CK/'
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    # start train
    trainer(valid_data=config['valid_data'])# valid_data: 1.5T or 3.0T




