
## Experiment Environment

```
OS: Ubuntu 16.04
CUDA: 10.2
GPU: Nvidia GTX 1080Ti x 1
```
## Build Virtual Environment (Using Conda)

```
None
```
## Model Architecture
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/%E6%9E%B6%E6%A7%8B%E5%9C%96.jpg'>

## Data Preparing
```
!pip install nibabel
!pip install scikit-image
!pip install torchio
!pip install albumentations
```
> * S1 must used 2d dimension. Change 3D nii or nii.gz data to 2D slices.
> * S2 must used 3d dimension.
> * Please refer to `None` build your dataset array or build random data array.
```
 S1 example(2D) [Sx[HxWxC]]:
 ./data_set/...
     1| 00001.nii -> 00001-01.nii, 00001-02.nii, 00001-03.nii, 00001-04.nii..., 00001-n.nii 
     2| 00002.nii -> 00002-01.nii, 00002-02.nii, 00002-03.nii, 00002-04.nii..., 00002-n.nii 
     3| 00003.nii -> 00003-01.nii, 00003-02.nii, 00003-03.nii, 00003-04.nii..., 00003-n.nii 
                  .
                  .
     N|     N.nii ->    ... 0000N-n.nii 
```
```
 S2 example(3D) [[HxWxSxC]]:
 ./data_set/...
     1| 00001.nii
     2| 00002.nii
     3| 00003.nii
                  .
                  .
     N|     N.nii 
```
```
concate data array-> [Sample Number, depth, width, height] ([140,32,384,384])
```

### Training S1 Semantic Segmentation Network
```
!python S1_Train.py
```
> * Edit env parameter value in `None`.

### Training S2 3D Classification Network

```
!python S2_Train.py
```
> * Edit env parameter value in `None`.

## Testing
> * `evaluate/S1_S2_test.ipynb`

## Perfromace Plot
> * None
