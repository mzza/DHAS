

The github repo is the code of paper "Deep Hierarchy-Aware Segmentation: A Novel Framework for MRIs Brain Tumor Segmentation" submitted to IEEE Transactions on Medical Imaging

If you face any problem, please feel free to open an issue.

## Directory structure

```
.
├─ data
  ├─ brats20	# Data provided by the BraTS 2020 competition host
    ├─ TrainingData
	├─ BraTS20_Training_001
	    ├─ BraTS20_Training_001_flair.nii.gz
	    ├─ BraTS20_Training_001_seg.nii.gz
	    ├─ BraTS20_Training_001_t1.nii.gz
	    ├─ BraTS20_Training_001_t1ce.nii.gz
	    ├─ BraTS20_Training_001_t2.nii.gz
	├─ BraTS20_Training_002
	├─ ...
    ├─ ValidationData
	├─ BraTS20_Validation_001
	    ├─ BraTS20_Validation_001_flair.nii.gz
	    ├─ BraTS20_Validation_001_t1.nii.gz
	    ├─ BraTS20_Validation_001_t1ce.nii.gz
	    ├─ BraTS20_Validation_001_t2.nii.gz
	    ├─ ...
├─ model	# Generated training and validation split (training_ids.pkl, validation_ids_pkl, test_ids.pkl), processed data file (brats20_data.h5, brats20_data_test.h5), and save best training model (isensee_2017_model.h5)
├─ output	# Generated prediction file
├─ src		# Souce code
    ├─ unet3d
    ├─ config.py
    ├─ inference.py
    ├─ train.py
├─ inference.sh
├─ train.sh
```

## Packages
- Python >= 3.5 (my current version is 3.7.7)
- tensorflowgpu==1.15 (other 1.x version should work)
- Other packages: pytables, SimpleITK, nilearn, nibabel
- Optional package: nipype (For n4itk bias correction preprocessing only. However, I didn't achieve that much performance gain using this technique!)

* **Note for installing 'pytables': install it using conda ('conda install pytables'). Installing using pip ('pip install tables') raises 'memory dump' issue!**

## How to run
If you prepare directory structure properly, you are done!

### Train:
~~~
$ ./train.sh
~~~
### Validation/Test:
~~~
$ ./inference.sh
~~~



## Acknowledgment
Significant code has been borrowed from [ellisdg's repository](https://github.com/ellisdg/3DUnetCNN) which is based on [Isensee et al.'s paper](https://doi.org/10.1007/978-3-030-11726-9_21). 
