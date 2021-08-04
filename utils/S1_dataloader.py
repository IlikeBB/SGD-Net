import numpy as np
# del black background
def del_darkimg(img,mask):
    temp_i = []
    temp_m = []
    for i in tqdm(range(img.shape[0])):
        if np.sum(img[i])!=0:
            temp_i.append(img[i])
            temp_m.append(mask[i])
    temp = np.array(temp_i)
    temp2 = np.array(temp_m)
    return temp, temp2

def dataloader(MRI_nii_folder_path, img_seg_path):
    # loading train data 3.0T: image / masks
    X_train = np.expand_dims(np.load(MRI_nii_folder_path + config["train_img_p3"]), axis=-1)
    y_train = np.expand_dims(np.load(MRI_nii_folder_path + config["train_msk_p3"]), axis=-1)
    # loading valida data 3.0T + 1.5T: image / masks
    X_valid = np.expand_dims(np.load(MRI_nii_folder_path + config["valid_img_p3"]), axis=-1)
    y_valid = np.expand_dims(np.load(MRI_nii_folder_path + config["valid_msk_p3"]), axis=-1)

    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    # (140, 32, 384, 384, 1)
    X_train = np.reshape(X_train, (X_train.shape[0]*32,384,384,1))
    y_train = np.reshape(y_train, (y_train.shape[0]*32,384,384,1))
    X_valid = np.reshape(X_valid, (X_valid.shape[0]*32,384,384,1))
    y_valid = np.reshape(y_valid, (y_valid.shape[0]*32,384,384,1))
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    X_train, y_train = del_darkimg(X_train, y_train)
    X_valid, y_valid = del_darkimg(X_valid, y_valid)
    print(X_train.shape, y_train.shape, X_valid.shape, y_valid.shape)
    return X_train.astype(np.float32), y_train.astype(np.int8), X_valid.astype(np.float32), y_valid.astype(np.int8)