## DESCRIPTION
This repository contains the PyTorch implementation of SpaNN, organized as follows:

directories:

cfg: contains cfg files for possible object detection victim models (YOLOv2 by default)
checkpoints: contains the weights for the AD attack detector net. The weights for ResNet50 on CIFAR-10 should be placed in this folder.
data: contains the folder structure required to run SpaNN on the datasets introduced in the paper. Due to the file size, only a small amount of examples are included for each dataset.
nets: torch models for AD and ResNet50
utils: contains utils.py, a file with helper functions of object detection.
weights: the weights for YOLOv2 should be placed in this folder.

files:

cfg.py, darknet.py, region_loss.py: files required to run YOLOv2 using PyTorch.
helper.py: various helper functions to perform object detection.
spann.py: main code to run and evaluate attack detection using SpaNN

Our code is based on the following publicly available repositories:

https://github.com/Zhang-Jack/adversarial_yolo2
https://github.com/inspire-group/PatchGuard/tree/master

For attacks on CIFAR-10 it is necessary to download the resnet50_192_cifar.pth file from https://github.com/inspire-group/PatchGuard/tree/master and place it in the checkpoints folder
For attacks on INRIA and Pascal VOC it is necessary to follow the instructions on https://github.com/Zhang-Jack/adversarial_yolo2 to download the yolo.weights file into the weights folder


![SpaNN (2)](https://github.com/user-attachments/assets/114d6bd4-be1d-42c8-83c6-4e032314b485)
Illustration of SpaNN: For any input $X$, after extracting a feature map $M$ from the shallow layers of the victim model $h$, a binarized feature map $B_b$ is obtained for each threshold $\beta_b$ in the set $\mathcal{B}$. DBSCAN is applied to each element in the ensemble, and the resulting clustering features $s$ are fed to the neural network $AD$, which outputs an attack detection score $AD(s)$.

## EXAMPLE COMMANDS
Run SpaNN on effective single-patch attacks on INRIA using default settings (with an attack detection threshold of 0.5):
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --effective_files effective_1p.npy --n_patches 1
```
For double patches:
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/2p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --effective_files effective_2p.npy --n_patches 2
```
To compute false alarm rates, use the clean counterpart of each attacked set by adding the "--clean" flag to the above commands (the "detected attacks" in the output refers to the false alarm rate), for example, for the effective single-patch attacks on INRIA:
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --effective_files effective_1p.npy --n_patches 1 --clean
```
To evaluate detection of non-effective attacks, add the "--uneffective" flag to the above commands, for example:
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --effective_files effective_1p.npy --n_patches 1 --uneffective
```
The commands presented so far take a default detection threshold of 0.5, to use a different threshold one can use the argument --nn_det_threshold, for example, to use a threshold of 0.1 in the first command:
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --effective_files effective_1p.npy --n_patches 1 --nn_det_threshold 0.1
```
The results in the paper involve tuning the detection threshold and computing ROC curves. To obtain the detection scores regardless of the threshold indicated by "--nn_det_threshold", use the "--save_scores" flag. To obtain detection scores for effective single-patch attacks on INRIA:
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_5_atk_det.pth --det_net 2dcnn_raw --ensemble_step 5 --effective_files effective_1p.npy --n_patches 1 --save_scores --savedir inria_resdir/
```
This will save the detection scores as a numpy array, in particular a .npy file containing the scores will be saved at the "inria_resdir/" directory. The same procedure can be followed to obtain scores corresponding to non-effective attacks, clean images, other patch attacks, etc.

The "--ensemble_step" argument refers to the size of the set of saliency thresholds used for attack detection. The maximum size of the set in the code is 100, which corresponds to a step of 1 (hence these parameters should be in the range 1-100).
Thus, a step 5 corresponds to the default 20 threshold set in the paper. To use, e.g., a set of size 50 run:
```
python spann.py --imgdir data/inria/clean --patch_imgdir data/inria/1p --dataset inria --det_net_path checkpoints/final_detection/2dcnn_raw_inria_2_atk_det.pth --det_net 2dcnn_raw --ensemble_step 2 --effective_files effective_1p.npy --n_patches 1
```
Note that the "ensemble_step" determines the size of the ensmeble used for detection, and thus it also indicates which saved weightfile must be used on the attack detector AD. The weightfile of AD will also depend on the dataset/task, for example, to run the SpaNN configuration from the command above on ImageNet, for attacks with four patches:
```
python spann.py --imgdir data/imagenet/clean --patch_imgdir data/imagenet/4p --dataset imagenet --det_net_path checkpoints/final_classification/2dcnn_raw_imagenet_10_atk_det.pth --det_net 2dcnn_raw --ensemble_step 10 --effective_files effective_1p.npy --n_patches 1
```
Refer to spann.py for further customization options.

## TRAINING
The weights for all trained instances of AD are available in this repository, however to further we also provide the training set feature maps we used to train AD (except the ones corresponding to Pascal VOC due to file size).

Example commands to train AD using the default ensemble size of 20 and are presented below:

Train AD for INRIA (YOLOv2 victim model):
```
python train_attack_detector.py --feature_maps net_train_data/inria/1p_train_fms.npy --adv_feature_maps net_train_data/inria/1p_train_pfms.npy
```
Train AD for ImageNet (ResNet-50 victim model):
```
python train_attack_detector.py --feature_maps net_train_data/imagenet/1p_train_fms.npy --adv_feature_maps net_train_data/imagenet/1p_train_pfms.npy
```
Train AD for CIFAR-10 (ResNet-50 victim model):
```
python train_attack_detector.py --feature_maps net_train_data/cifar/1p_train_fms.npy --adv_feature_maps net_train_data/cifar/1p_train_pfms.npy
```

Weights will be saved to the checkpoints folder.
