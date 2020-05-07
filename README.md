# Deep Blind Video Super-resolution(DBVSR)

This repository is an official PyTorch implementation of the paper "[Deep Blind Video Super-resolution](https://arxiv.org/abs/2003.04716v1)".  
The code is built on Ubuntu 16.04 environment (Python3.6, PyTorch_0.4.1, CUDA8.0, cuDNN5.1) with Tesla V100/1080Ti GPUs.


## Dependencies
* ubuntu16.04
* python 3.6(Recommend to use Anaconda)
* pyTorch0.4.1
* numpy
* skimage
* imageio
* matplotlib
* tqdm
* cv2 

## Get started

#### Trainset:
We use the REDS dataset to train our models. You can download it from [official website](https://seungjunnah.github.io/Datasets/reds.html)  

We regroup the REDS training and validation sets same as [EDVR](https://github.com/xinntao/EDVR) do:  
trainset: the original training (except 4 clips) and validation sets, total 266 clips  
validationset: 000, 011, 015 and 020 clips from the original training set, total 4 clips

#### Models
All the models(X2, X3, X4) can be downloaded from [GoogleDrive](https://drive.google.com/drive/folders/1qqJzwbA_Xwsv_yOGuglUezaY2SduhupR?usp=sharing).

## Quicktest with benchmark
After download our models in paper, place the folder ``models_in_paper`` to the path ``./DBVSR``
You can test our super-resolution algorithm with REDS4 dataset. Please organize the testset in  ``testset`` folder like this:  
```
        |--REDS
           |--test
              |--HR
                  |--000
                      |--00000000.png
                             :
                             :
                      |--00000099.png
                  |--011
                  |--015
                  |--020
              |--LR
                  |--000
                      |--00000000.png
                             :
                             :
                      |--00000099.png
                  |--011
                  |--015
                  |--020
```
please check the data root of test sets in code ``./code/option/template.py``, line 9(args.dir_data_test).(for dbvsr)   
please check the data root of test sets in code ``./code/option/template.py``, line 26(args.dir_data_test).(for baseline_lr)    
please check the data root of test sets in code ``./code/option/template.py``, line 43(args.dir_data_test).(for baseline_hr)    
    
Then, run the following commands:
```bash
cd code
python main.py --test_only
```

And generated results can be found in ``./experiment/dbvsr_test/results/`` for dbvsr results  
And generated results can be found in ``./experiment/baseline_lr_test/results/`` for baseline_lr results  
And generated results can be found in ``./experiment/baeline_hr_test/results/`` for baseline_hr results  
  * To test other benchmarks, you can modify the option(dir_data_test) of the command above.   
  * To change the save root, you can modify the option(save) of the command above.  
  

## How to train
If you have downloaded the trainset, please make sure that the trainset has been organized as follows:
```
       |--REDS
           |--train
              |--HR
                  |--001
                      |--00000000.png
                      |--00000001.png
                             :
                             :
                      |--00000099.png
                  |--002
                      :
                      :
                  |--239
              |--LR
                  |--001
                      |--00000000.png
                      |--00000001.png
                             :
                             :
                      |--00000099.png
                  |--002
                      :
                      :
                  |--239
```
Then, 
please check the data root of train sets in code ``./code/option/template.py``, line 6(args.dir_data).(for dbvsr)  
please check the data root of train sets in code ``./code/option/template.py``, line 23(args.dir_data).(for baseline_lr)  
please check the data root of train sets in code ``./code/option/template.py``, line 40(args.dir_data).(for baseline_hr)  

The command for training is as follow:
```
cd code
python main.py
```

The pretrain_model of pwc-net and fcnet can be found in ``./pretrain``.
