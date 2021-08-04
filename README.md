<h2>Experiment Environment</h2>

```text
OS: Ubuntu 16.04
CUDA: 10.2
GPU: Nvidia GTX 1080Ti x 1
```

<h2>Build Virtual Environment(Using Conda) </h2>

```text
!conda env create -f MRI_tf2.yml

!conda activate MRI_tf2
```
<h2>Training</h2>
<b><h3>Training S1 Semantic Segmentation Network</h3></p>
Edit env parameter value in /utils/model_config_S1.yaml

```text
!python 01.S1_Training.py
```
<b><h3>Training S2 3D Classification Network</h3></p>
Edit env parameter value in /utils/model_config_NL or AP.yaml

```text
!python 02.S2_AP_Training.py

!python 02.S2_NL_Training.py
```
<h2>Testing</h2>

<h2>Perfromace Plot</h2>

<h2>Visual Results</h2>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0316.gif'></p>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0319.gif'></p>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0342.gif'></p>
<img src='https://github.com/IlikeBB/F3DD/blob/main/plot_results/is0337.gif'></p>

<h3>Reference</h3>
<a href='https://github.com/albumentations-team/albumentations'> Albumentations</a></p>
<a href='https://github.com/mjkvaak/ImageDataAugmentor'> ImageDataAugmentor</a></p>
<a href='https://github.com/JihongJu/keras-resnet3d'> 3D ResNet</a></p>
<a href='https://github.com/qubvel/segmentation_models'> Segmentation Models</a></p>
