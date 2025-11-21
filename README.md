# Absolute-Unified Rehabilitation Exercise Assessment

Official implementation of the paper "Absolute-Unified Rehabilitation Exercise Assessment using a Multiscale Spatio-Temporal Graph Convolutional Network".


## Requirements

This project is developed with `Python 3.8.10` and the following packages:

```
addict==2.4.0
cycler==0.12.1
dill==0.4.0
einops==0.7.0
fonttools==4.54.1
joblib==1.4.2
kiwisolver==1.4.7
markdown-it-py==3.0.0
matplotlib==3.5.1
mdurl==0.1.2
numpy==1.22.4
opencv-python==4.5.4.60
packaging==24.1
pandas==1.5.2
pillow==10.4.0
platformdirs==4.3.6
Pygments==2.18.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0
rich==13.9.4
scikit-learn==1.0.2
scipy==1.10.1
seaborn==0.11.2
six==1.16.0
termcolor==2.4.0
threadpoolctl==3.5.0
tomli==2.2.1
torch==1.11.0+cu113
typing_extensions==4.12.2
tzdata==2024.2
yapf==0.43.0
```

## Dataset

We conducted the experiment using the following publicly available dataset: IRDS, UI-PRMD, and REHAB24-6.

1. IRDS: [download link](https://zenodo.org/record/4610859#.Y2Z4kctBxJB). 

2. UI-PRMD: [download link](https://opendatalab.com/OpenDataLab/UI-PRMD/tree/main). (Its official download link `https://www.webpages.uidaho.edu/ui-prmd/` is no longer available.)

3. REHAB24-6: [download link](https://zenodo.org/records/13305826).

After downloading these datasets, organize them according to the following structure:

```
|-- data
    ├── IRDS
        └── Simplified
            |-- xx.txt
            |-- ...
    ├── REHAB24-6
    │   └── 3d_joints_segmented
            |-- xx.txt
    └── UI-PRMD
        ├── Correct
            ├── Kinect
                |-- Angles
                |-- Positions
                |-- Skeletons
                    |-- xx.txt
                    |-- ...
            └── Vicon
        └── Incorrect
            ├── Kinect
            ├── Vicon
```

You may need to preprocess the UI-PRMD and REHAB24-6 datasets separately using `datasets/preprocess.py` and `datasets/preprocess_rehab24_6.py`, respectively.

## Test Model

1. You can directly download the trained models from [here](https://1drv.ms/f/c/d94b0946a1e8e2cc/IgABGedUTsaRR45pKSWUabpQAR000O6yzvI8kcgnG4Rkr1w). 

2. Place the trained models in the main directory of the project.

3. Use the following script for testing:

```
IRDS:
python inference.py --model msst-gcn --dataset IRDS --inferdir inference_dir_name --num_classes 2 --unified

UI-PRMD (Kinect):
python inference.py --model msst-gcn --dataset UI-PRMD_Kinect --inferdir inference_dir_name --num_classes 2 --unified


UI-PRMD (Vicon):
python inference.py --model msst-gcn --dataset UI-PRMD_Vicon --inferdir inference_dir_name --num_classes 2 --unified


REHAB24-6:
python inference.py --model msst-gcn --dataset REHAB24-6 --inferdir inference_dir_name --num_classes 2 --unified

```

## References

This project is inspired by the following works:

1.Zheng, Kaili, et al. "A skeleton-based rehabilitation exercise assessment system with rotation invariance." IEEE Transactions on Neural Systems and Rehabilitation Engineering 31 (2023): 2612-2621.

2.Myung, Woomin, et al. "Degcn: Deformable graph convolutional networks for skeleton-based action recognition." IEEE Transactions on Image Processing 33 (2024): 2477-2490.

