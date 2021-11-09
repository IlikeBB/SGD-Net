import os
import nibabel as nib
from skimage import morphology
from scipy import ndimage
from PIL import Image
import numpy as np
depth = 32
class nii_process:
    def __init__(self):
        self.depth, self.size = 32, 384
        self.path, self.nii_name = '',''
        self.volume = np.array([])

    def normalize(self, volume):
        img_o = np.float32(volume.copy())
        m = np.mean(img_o)
        s = np.std(img_o)
        volume = np.divide((img_o - m), s)
        self.image = volume.astype("float32")

    def process_scan(self):
        image = nib.load(self.path)
        affine = image.header.get_best_affine()

        image = image.get_fdata()
        # print(image.shape)
        self.slice_n = image.shape[-1]
        if affine[1, 1] > 0:
            self.image = ndimage.rotate(image, 90, reshape=False, mode="nearest")
        if affine[1, 1] < 0:
            self.image = ndimage.rotate(image, -90, reshape=False, mode="nearest")

        self.normalize(self.image)
        if affine[0, 0] < 0:
            for i in range(self.volume.shape[0]):
                self.volume[i,:,:] = np.fliplr(self.volume[i,:,:])

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
        adjusted_seg.to_filename(os.path.join(self.base_ , save_folder,f'{self.nii_name[0:-7]}_pred_lesion.nii.gz'))  

if __name__ == '__main__':
    if not os.path.exists('./checkpoint'):
        os.makedirs('./checkpoint')
    save_folder = './'
    pass