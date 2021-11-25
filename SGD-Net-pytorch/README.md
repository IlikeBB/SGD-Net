
## Experiment Environment

```
OS: Ubuntu 18.06
CUDA: 10.2
GPU: Nvidia GTX 1080Ti x 1
Python: 3.8.0
```
## Build Virtual Environment (Using Conda)

```
!pip install torch==1.9.1 -y
!pip install torchio==0.18.62 -y
!pip install segmentation_models_pytorch
!pip install albumentations==1.1.0 -y
!pip install thop
```
## Model Architecture
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/%E6%9E%B6%E6%A7%8B%E5%9C%96.jpg'>

## Data Preparing
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
> * If use CGRD dataset, please modify torchio loader code  `{env_name}/../torchio/data/io.py` .
```
def read_image(path: TypePath) -> Tuple[torch.Tensor, np.ndarray]:
    try:
        result = _read_nibabel(path)
    except nib.loadsave.ImageFileError as e:
        message = (
            f'File "{path}" not understood.'
            ' Check supported formats by at'
            ' https://simpleitk.readthedocs.io/en/master/IO.html#images'
            ' and https://nipy.org/nibabel/api.html#file-formats'
        )
        raise RuntimeError(message) from e
    return result
```
```
def _read_nibabel(path: TypePath) -> Tuple[torch.Tensor, np.ndarray]:
    from scipy import ndimage
    img = nib.load(str(path))
    affine = img.header.get_best_affine()
    data = img.get_fdata(dtype=np.float32)
    data = check_uint_to_int(data)
    if affine[1, 1] > 0:
        data = ndimage.rotate(data, 90, reshape=False, mode="nearest")
    if affine[1, 1] < 0:
        data = ndimage.rotate(data, -90, reshape=False, mode="nearest")
    if affine[1, 1] < 0:                 
        data = np.fliplr(data)    
    tensor = torch.as_tensor(data.copy())
    return tensor, affine
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
```
ap  -   Accuracy  : 91.0 %
ap  -   Sensitivity  : 0.71429
ap  -   Specificity  : 1.0
nl  -   Accuracy  : 93.0 %
nl  -   Sensitivity  : 0.95833
nl  -   Specificity  : 0.90476
```
## Perfromace Plot
> * None
