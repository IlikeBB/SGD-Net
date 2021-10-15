# ------------------------model import------------------------
import os, sys, numpy as np, tensorflow as tf, shutil, dicom2nifti
import segmentation_models as sm
from model.resnet3d import Resnet3DBuilder
from segmentation_models import Unet
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import nibabel as nib
from skimage import morphology
from scipy import ndimage
from PIL import Image
import numpy as np
depth = 32
threshold = 0.5
S1_weight = "./model_weight/S1_DenseNet.hdf5"
S2_weight_AP = "./model_weight/S2_ResNet50_AP.hdf5"
S2_weight_NL = "./model_weight/S2_ResNet50_NL.hdf5"
class nii_process:
    def __init__(self, model_weight):
        self.depth, self.size = 32, 384
        self.S2_ans = [2,2] #NL(0,1), AP(0,1)  P.S.    *2 = Negative*
        self.path, self.nii_name = '',''
        self.volume = np.array([])
        self.model = Unet('densenet121', encoder_weights=None, input_shape=(None, None, 1))
        self.model2 = Resnet3DBuilder.build_resnet_50((self.depth, self.size, self.size, 1), 1)
        self.model.load_weights(model_weight)
    def normalize(self, volume, norm_type):
        if norm_type == 'zero_mean':
            img_o = np.float32(volume.copy())
            m = np.mean(img_o)
            s = np.std(img_o)
            volume = np.divide((img_o - m), s)

        elif norm_type == 'div_by_max':
            volume = np.divide(volume, np.percentile(volume,98))
            
        elif norm_type == 'onezero':
            for channel in range(volume.shape[-1]):
                volume_temp = volume[..., channel]
                volume_temp = (volume_temp - np.min(volume_temp)) / (np.max(volume_temp)-np.min(volume_temp))

                volume[..., channel] = volume_temp
        self.image = volume.astype("float32")

    def resize_volume(self):
        """Resize across z-axis"""
        # Set the desired depth
        current_depth = self.image.shape[-1]
        current_width = self.image.shape[0]
        current_height = self.image.shape[1]
        self.volume = ndimage.zoom(self.image, (self.size/current_height, self.size/current_width, 1), order=0)
        # return self.image
    
    def process_scan(self):
        image = nib.load(self.path)
        affine = image.header.get_best_affine()

        if len(image.shape) == 4:
            print('True')
            image = image.get_fdata()
            width,height,queue,_ = image.shape
            image = image[:,:,:,1]
            image = np.reshape(image,(width,height,queue))
            adjusted_dwi = nib.Nifti1Image(image.astype(np.uint16), affine)
            os.remove(self.path)
            adjusted_dwi.to_filename(self.path)  
        else:
            image = image.get_fdata()
            pass
        # print(image.shape)
        self.slice_n = image.shape[-1]
        if affine[1, 1] > 0:
            self.image = ndimage.rotate(image, 90, reshape=False, mode="nearest")
        if affine[1, 1] < 0:
            self.image = ndimage.rotate(image, -90, reshape=False, mode="nearest")

        self.normalize(self.image, "zero_mean")
        self.resize_volume()
        #   add only black background mri image
        if self.volume.shape[2]!=self.depth:
            add_black_num = self.depth - self.volume.shape[2]
            self.volume = self.volume.transpose(2,0,1)
            for i in range(add_black_num):
                add_black_ = np.expand_dims(np.zeros((self.volume.shape[2],self.volume.shape[2])),axis=0)
                self.volume = np.concatenate((self.volume, add_black_), axis = 0)
            self.volume = self.volume.transpose(1,2,0)
        self.volume = self.volume.transpose(2,0,1)
        # print(volume.shape)
        if affine[0, 0] < 0:
            for i in range(self.volume.shape[0]):
                self.volume[i,:,:] = np.fliplr(self.volume[i,:,:])
    
    def predict_2(self):
        pred_list = [S2_weight_NL, S2_weight_AP]
        
        for idx,i in enumerate(pred_list):
            self.model2.load_weights(i)
            print(self.S2_dwi_npy.shape)
            S2_pred_X = np.expand_dims(self.S2_dwi_npy, axis=0)
            results= self.model2.predict(S2_pred_X , batch_size=1, verbose=1)
            self.S2_ans[idx]  = results[0]

    def predict(self, path, nii_name, lesion_name, threshold):
        self.base_ = path
        self.path = os.path.join(path, lesion_name)
        self.nii_name = nii_name
        self.process_scan()
        dwi_img = self.volume
        
        dwi_npy = np.reshape(dwi_img, (self.depth, self.size, self.size,1))
        S1_results = self.model.predict(dwi_npy, batch_size=1, verbose=1)
        for i in range(dwi_npy.shape[0]):
            dwi_npy[i,:,:] = np.fliplr(dwi_npy[i,:,:])
        if np.sum(S1_results)>0:
            self.S2_dwi_npy = np.where(S1_results > threshold, dwi_npy, dwi_npy*0)
            self.predict_2()
            print(self.S2_ans)
            self.S2_ans = [(i > threshold).astype(np.int8) for i in self.S2_ans]
            print(self.S2_ans)
        S1_pred_seg = np.squeeze(np.array([S1_results>threshold]), axis=0)
        S1_pred_seg = np.squeeze(S1_pred_seg[0:self.slice_n], axis=-1)
        dwi_npy = np.squeeze(dwi_npy[0:self.slice_n], axis=-1)

        print(S1_pred_seg.shape)
        self.get_pred_nii(S1_pred_seg)
        return  dwi_npy, S1_pred_seg, self.S2_ans

    def get_pred_nii(self, pred_img):
        original_img = nib.load(self.path)
        affine = original_img.header.get_best_affine()
        # fliplr pred seg
        if affine[0, 0] < 0:
            for i in range(pred_img.shape[0]):
                pred_img[i,:,:] = np.fliplr(pred_img[i,:,:])
            pred_img = pred_img.transpose(1,2,0)
        # change affine position
        if affine[1, 1] > 0:
            pred_img = ndimage.rotate(pred_img, -90, reshape=False, mode="nearest")
        if affine[1, 1] < 0:
            pred_img = ndimage.rotate(pred_img, 90, reshape=False, mode="nearest")
        adjusted_seg = nib.Nifti1Image(pred_img.astype(np.uint16), affine)
        adjusted_seg.header['pixdim'] = original_img.header['pixdim']
        # Save as NiBabel file
        adjusted_seg.to_filename(os.path.join(self.base_ , f'{self.nii_name}_pred_lesion.nii.gz'))  

dwi_process = nii_process(S1_weight)
def main(path):
    split_path =path.split('/')
    base_path = './'
    dicom_patient = os.listdir(path)
    # if not os.path.exists(os.path.join(base_path, 'Patient', split_path[-1],'nii')):
    #     os.makedirs(os.path.join(base_path, "Patient", split_path[-1],'nii'))
    # print(path)
    # dicom2nifti.convert_directory(path, os.path.join(base_path, "Patient", split_path[-1],'nii'))
    # print('pass')
    for x in os.listdir(os.path.join(base_path, "Patient", split_path[-1],'nii')):
        if 'tracew.nii.gz' in x:
            dwi_ = x
        elif 'mprage' in x:
            t1_ = x
        else:
            os.remove(os.path.join(base_path, "Patient", split_path[-1],'nii',x))
#         # ------------------------------S1 Lesion pred-------------------------
    print(dwi_, t1_)
    img_name =split_path[-1]
    print('\n\n')
    dwi_img, pred_mask, S2_ans = dwi_process.predict(os.path.join(base_path, "Patient", split_path[-1],'nii'), img_name, dwi_, threshold)
    # return 0, 0
    LN_list = ['Non-Lacune', 'Lacune', 'Negative']
    AP_list = ['Anterior', 'Posterior', 'Negative']
    LN = LN_list [S2_ans[0][0]]
    AP = AP_list[S2_ans[1][0]] 
    np.save('./dwi_img',dwi_img)
    np.save('./pred_mask', pred_mask)
    return dwi_img, pred_mask, split_path[-1], LN, AP

if __name__ == '__main__':
    base_path = './'
    reg_folder_name, img_folder = 'results_patient', 'test_data'
    dicom_patient = os.listdir("./dataset/")
    for i in dicom_patient:
        for j in (os.listdir(os.path.join('dataset',i))):
            if not os.path.exists(os.path.join(base_path, 'Patient', i,'nii')):
                os.makedirs(os.path.join(base_path, "Patient", i,'nii'))
            print(os.path.join(base_path, "dataset",i,j))
            dicom2nifti.convert_directory(os.path.join(base_path, "dataset",i,j), os.path.join(base_path, "Patient", i,'nii'))

        for j in (os.listdir(os.path.join('dataset',i))):
            for x in os.listdir(os.path.join(base_path, "Patient", i,'nii')):
                if 'tracew.nii.gz' in x:
                    dwi_ = x
                elif 'mprage' in x:
                    t1_ = x
                else:
                    os.remove(os.path.join(base_path, "Patient", i,'nii',x))
        # ------------------------------S1 Lesion pred-------------------------
        patient_name = i
        img_name = j
        print('\n\n')
        dwi_process.predict(os.path.join(base_path, 'Patient',i , 'nii'), patient_name, img_name, dwi_, threshold)
        # ------------------------------Atlas---------------------------------------
        next_path = os.path.join(base_path, img_folder,'nii', patient_name, img_name)
        data_stack = [f'{next_path}/{dwi_}', f'{next_path}/{t1_}', f'{next_path}/{patient_name}_pred_lesion.nii.gz']
        print(data_stack)
        # # diff = DWI, T1 = Mprage data, lesion = Mask Label
        # os.system('bash MNI_process_iter_0917.sh '+ ' '+base_path+' '+reg_folder_name+ ' '+data_stack[0]+' '+ data_stack[1] + ' ' + data_stack[2] + ' ' + patient_name + ' ' +img_name)
        