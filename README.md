CullNet
=======

This code implements the following paper:

["CullNet: Calibrated and Pose Aware Confidence Scores for Object Pose Estimation"](https://arxiv.org/pdf/1909.13476.pdf),
Kartik Gupta, Lars Petersson, and Richard Hartley,
ICCV Workshop on Recovering 6D Object Pose, 2019.

This code is for research purposes only.

Any questions or discussions are welcomed!

If you're using this code, please cite our paper.


## Installation

Setup python 2.7.15 environment
```
pip install -r requirements.txt
```

## Dataset Configuration

### Prepare the dataset

1. Create data directory.

    ```
    mkdir data
    cd data
    ```

2. Download the LINEMOD, which can be found at [here](https://1drv.ms/u/s!AtZjYZ01QjphgQ56t4wCharVSfxL).

3. Download the LINEMOD_ORIG, which can be found at [here](./tools/download_linemod_orig.sh).

4. Generate synthetic images for each class using [pvnet-rendering](https://github.com/zju-3dv/pvnet-rendering) by the following command in the repo. Note, generated data in LINEMOD should be in data directory of this repo.

    ```
    python run.py --type rendering
    ```
5. Download the following PASCAL VOC images for background and LINEMOD data.

    ```
    Download LINEMOD_original from  "https://drive.google.com/open?id=1uhv5bvm3XQ6Zsx8vOkFXfbxVrIIkmfLf"
    Download Occluded_LINEMOD from "https://cloudstore.zih.tu-dresden.de/index.php/s/a65ec05fedd4890ae8ced82dfcf92ad8/download"
    wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
    tar xf LINEMOD_original.tar
    tar xf VOCtrainval_11-May-2012.tar
    ```

6. Create training dataset (in the trainloader format) as combination of blender based 10000 synthetic images (for each class) generated above and rest augmented real training images.

    ```
    cd data_generate
    python createdataset_singleobj_realsyn.py
    python createdataset_singleobj_occlusiontest.py
    python createdataset_singleobj_originalimages.py
    ```
7. Download the pre-trained Yolov3 detector (for object detection) and put it in `models/trained_models`

   ```
   wget -c https://pjreddie.com/media/files/darknet53.conv.74
   ```

## Training and testing

### Training

Backbone training (Yolov3-6D):
```
class='cam'
gpu_id=0
CUDA_VISIBLE_DEVICES=$gpu_id python yolo6d_v3_train.py --lr 0.001 --start_epoch 0 --epochs 50 --exp_name yolo6d_v3_realsyn_$class  --bs 16 --torch_cudnn_benchmark --class_name $class --use_tensorboard  --datadirectory_name LINEMOD_singleobj_realsyn --diff_aspectratio --random_trans_scale
```
CullNet training:
``` 
cls='cam'
gpu_id=0
CUDA_VISIBLE_DEVICES=$gpu_id python cullnet_train.py --lr 0.01 --epochs 116 --exp_name yolo6d_v3_realsyn_$cls --bs 16 --exp_name_cullnet lr01_resnet50_gn_concat_synreal_subbs128_steplrSGD --cullnet_type resnet50concat_gn --cullnet_input 112 --seg_cullnet --cullnet_inconf concat --cullnet_confidence conf2d --thresh 0.1 --non_nms --torch_cudnn_benchmark --class_name $cls --use_tensorboard --datadirectory_name LINEMOD_singleobj_realsyn --diff_aspectratio --epoch_save --start_epoch 99 --k_proposals 32 --cullnet_trainloader all --sub_bs 128 --sub_bs_test 16
```

### Testing

We provide the pretrained models of each object, which can be found at [here](https://drive.google.com/open?id=1OQ6Fn26FJrvJoz_lN_RJFbXudQ-cmlrN).
Download the pretrained models and move them to `models/`.

```
cls='cam'
gpu_id=0
CUDA_VISIBLE_DEVICES=$gpu_id python -W ignore cullnet_test.py --exp_name yolo6d_v3_realsyn_$cls --exp_name_cullnet lr01_resnet50_gn_concat_synreal_subbs128_steplrSGD --bs 1 --class_name $cls --cullnet_input 112 --cullnet_inconf concat --datadirectory_name LINEMOD_singleobj_realsyn --diff_aspectratio --image_size_index 3 --thresh 0.1 --non_nms --seg_cullnet --cullnet_type resnet50concat_gn --torch_cudnn_benchmark  --cullnet_confidence conf2d --ADI --nearby_test 1 --sub_bs_test 6 --topk 6 --epoch_save --start_epoch 115 --save_results --bias_correction mode
```

#### Acknowledgments
This code is written by [Kartik Gupta](https://cecs.anu.edu.au/people/kartik-gupta) and is built on the YOLOv2 and YOLOv3 implementations of the github user [@longcw](https://github.com/longcw) and [@eriklindernoren](https://github.com/eriklindernoren). For training, the synthetic data has been generated using the blender scripts of github user [@zju3dv](https://github.com/zju3dv). 


#### Contact
Kartik Gupta (kartik.gupta@anu.edu.au).

