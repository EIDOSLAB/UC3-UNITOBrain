# UC3 UNITOBrain 

This repository contains the source code for getting started on [UniToBrain](https://ieee-dataport.org/open-access/unitobrain) dataset by using [pyEDDL](https://github.com/deephealthproject/pyeddl)/[pyECVL](https://github.com/deephealthproject/pyecvl)

# Requirements
* numpy
* pyeddl
* pyecvl
* cv2
* wandb
* scipy
* pydicom

# 1) Preprocessing Run Command
The network inputs are tensors of multiple CT images at the same scansion height for each patient.
This script creates a folder called `input_tensored` into the path 'prep_output_path' path.

Image size of `128` pixels is used for the pratrain phase, size of `512` for the full resolution training  

```
# 1.1) compute the inputs
python3 -u dicomtopt.py --rescale_size 128 --prep_output_path <output-path> --unitobrain_path <unitobrain-path> --target 'INPUT'

# 1.2) compute the target perfusion maps
python3 -u dicomtopt.py --rescale_size 128 --prep_output_path <output-path> --unitobrain_path <unitobrain-path> --target 'TTP'
python3 -u dicomtopt.py --rescale_size 128 --prep_output_path <output-path> --unitobrain_path <unitobrain-path> --target 'CBF'
python3 -u dicomtopt.py --rescale_size 128 --prep_output_path <output-path> --unitobrain_path <unitobrain-path> --target 'CBV'
```

# 2) Pretrain Run Command (Optional)
Pretrained model on lower resolution tensors (target TTP on 4 gpus)
```
python3 -u train_model.py --target 'TTP' --shape 128 --lr 1e-5 --num_gpu 4 --epochs 100 --batch-size 8 --mem 'low_mem' --name <run-name> <prep_output_path>
``` 
  
# 3) Train Run Command
Train the model on full resolution tensors (target TTP on 4 gpus)
```
python3 -u train_model.py --target 'TTP' --resume_ckpts <pretrain-checkpoint> --batch-size 8 --batch-size-val 4 --lr 1e-5 --epochs 50 --num_gpu 4 --name <run-name> --mem 'low_mem' --shape 512 --log-interval 20 <prep_output_path>
``` 

# 4) Inference Run Command
Inference command to run Tests only
```
python3 -u test_model.py --target 'TTP' --shape 512 --gpu 4 --mem 'low_mem' --ckpts <train-checkpoint> <prep_output_path>
``` 

