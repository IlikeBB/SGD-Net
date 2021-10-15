import numpy as np, os
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.colors 
from scipy import ndimage

def MNI_process(path):
    images = nib.load(path)
    affine = images.header.get_best_affine()
    images = images.get_fdata()
    images = ndimage.rotate(images, 90, reshape=False, mode="nearest")
    images = images.transpose(2,0,1)
    if affine[0, 0] > 0:
        for i in range(images.shape[0]):
            images[i,:,:] = np.fliplr(images[i,:,:])
    return images

def main2(patient_name):
# def main2():
    # print(patient_name)
    MNI_path = os.path.abspath(os.getcwd())+'/MNIs/'
    print(MNI_path)
    # patient_name = 'B8DC3364FE2452484459251ACFE2C4336AE4A4D0' #patient paramenter value
    MNI_save_path = f'./Patient/{patient_name}/atlas/'
    
    JHU_atlas = MNI_process('./MNIs/JHU_MNI.nii.gz')
    JUELICH_atlas = MNI_process('./MNIs/JUELICH_25_MNI.nii.gz')
    BA_atlas = MNI_process('./MNIs/BA_MNI.nii.gz')
    AAL_atlas = MNI_process('./MNIs/AAL3_MNI.nii.gz')
    # Atlas extract
    next_path = os.path.join('Patient', patient_name,'nii')
    for i in os.listdir('./'+next_path):
        if 'pred' in i:
            pred_nii = i
        elif 'tracew' in i:
            dwi_nii = i
        elif 'mprage' in i:
            ti_nii = i

    nii_path =  os.path.join(os.path.abspath(os.getcwd()),next_path) #t1, dwi, lesion path
    print(nii_path)
    data_stack = [f'{nii_path}/{dwi_nii}', f'{nii_path}/{ti_nii}', f'{nii_path}/{pred_nii}']
    #                       dwi_path, t1_path, pred_path
    check_data = os.listdir(f'./Patient/{patient_name}/atlas/')
    if 'lesion2MNI.nii.gz' in str(check_data):
        pass
    else:
        os.system('bash ./MNIs/MNI_process_iter_0917.sh' + ' '+MNI_path+' '+MNI_save_path + ' '+
                                data_stack[0]+' '+ data_stack[1] + ' ' + data_stack[2] + ' ' + patient_name)
    
    MNI_Lesion = MNI_process(f'./Patient/{patient_name}/atlas/lesion2MNI.nii.gz')
    print('pass')
    MNI_Atlas_stack = [AAL_atlas, BA_atlas, JHU_atlas, JUELICH_atlas]

    # get atlas csv values
    csv_path = f'./Patient/{patient_name}/atlas/LesionMapping_{patient_name}.csv'
    main_csv = np.array(pd.read_csv(csv_path, header=None))
    MNI_Atlas_sorted = {'AAL': [], 'BA':[], 'JHU':[], 'JUELICH':[]}
    for i in MNI_Atlas_sorted:
        for j in main_csv:
            if str(i) in str(j[0]):
                MNI_Atlas_sorted[i].append(j)

    

    return MNI_Atlas_stack, MNI_Lesion, MNI_Atlas_sorted

if __name__ == '__main__':
    main2()
    # for i in range(1, atlas.shape[0]):
    #     plt.subplot(9,10,i)
    #     if (i-1)>atlas.shape[0]:
    #         blank = np.ones((91,109))
    #         plt.imshow(blank, cmap='bone')
    #     else:
    #         plt.imshow(atlas[i-1], alpha= 0.7, cmap = matplotlib.colors.ListedColormap(['lightblue' ,'gray', 'black']))
    #         plt.imshow(lesion[i-1]*150, alpha=0.9, cmap = matplotlib.colors.ListedColormap(['None' ,'black', 'red']))
            
            
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()