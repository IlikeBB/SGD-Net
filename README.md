## Experiment Environment

```
OS: Ubuntu 16.04
CUDA: 10.2
GPU: Nvidia GTX 1080Ti x 1
```
## Build Virtual Environment(Using Conda)

```
!conda env create -f MRI_tf2.yml

!conda activate MRI_tf2
```
## Model Architecture
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/%E6%9E%B6%E6%A7%8B%E5%9C%96.jpg'>

## Data Preparing
```
!pip install nibabel
!pip install scikit-image
```
> * Loading nii or nii.gz data and transformation to numpy data array.
> * Please refer to `nii data loading.ipynb` build your dataset array or build random data array.
```
 example:
 ./data_set/...
     1| 00001.nii -> 00001.npy
     2| 00002.nii -> 00002.npy
     3| 00003.nii -> 00003.npy
                  .
                  .
     n|     n.nii ->     n.npy
```
```
concate data array-> [Sample Number, depth, width, height] ([140,32,384,384])
```

## Training
### Training S1 Semantic Segmentation Network
```
!python 01.S1_Training.py
```
> * Edit env parameter value in `/utils/model_config_S1.yam`.

### Training S2 3D Classification Network
```
!python 02.S2_AP_Training.py

!python 02.S2_NL_Training.py
```
> * Edit env parameter value in `/utils/model_config_NL or AP.yaml`.

## Testing
```
pip install -U scikit-learn
```
> * Please refer to `03.evaluate.ipynb`
> * Edit env parameter value in `/utils/model_config.yaml`.

## Perfromace Plot
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/fig4-5%20revise%20table.001.png'>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/fig4-5%20revise%20table.002.png'>

## Visual Results
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0309.gif'></p>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0316.gif'></p>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0323.gif'></p>

## Reference
#### Data Augmentation reference from<a href='https://github.com/albumentations-team/albumentations'> Albumentations</a>, <a href='https://github.com/mjkvaak/ImageDataAugmentor'> ImageDataAugmentor</a>.
#### The code is heavily adapted from<a href='https://github.com/JihongJu/keras-resnet3d'> 3D ResNet</a>, <a href='https://github.com/qubvel/segmentation_models'> Segmentation Models</a>.
