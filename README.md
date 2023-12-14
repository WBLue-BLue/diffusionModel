# MedSegDiff: Medical Image Segmentation with Diffusion Model
MedSegDiff a Diffusion Probabilistic Model (DPM) based framework for Medical Image Segmentation. The algorithm is elaborated in our paper [MedSegDiff: Medical Image Segmentation with Diffusion Probabilistic Model](https://arxiv.org/abs/2211.00611) and [MedSegDiff-V2: Diffusion based Medical Image Segmentation with Transformer](https://arxiv.org/pdf/2301.11798.pdf).
## Requirement

``pip install -r requirement.txt``

## Example Cases
### Melanoma Segmentation from Skin Images
1. Download ISIC dataset from https://challenge.isic-archive.com/data/. Your dataset folder under "data_dir" should be like:

ISIC/

     ISBI2016_ISIC_Part3B_Test_Data/...
     
     ISBI2016_ISIC_Part3B_Training_Data/...
     
     ISBI2016_ISIC_Part3B_Test_GroundTruth.csv
     
     ISBI2016_ISIC_Part3B_Training_GroundTruth.csv
    
2. For training, run: ``python scripts/segmentation_train.py --data_name ISIC --data_dir input data direction --out_dir output data direction --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --lr 1e-4 --batch_size 8``
    
3. For sampling, run: ``python scripts/segmentation_sample.py --data_name ISIC --data_dir input data direction --out_dir output data direction --model_path saved model --image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 --diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False --num_ensemble 5``

4. For evaluation, run ``python scripts/segmentation_env.py --inp_pth *folder you save prediction images* --out_pth *folder you save ground truth images*``


In default, the samples will be saved at `` ./results/``
## Suggestions for Hyperparameters and Training
To train a fine model, i.e., MedSegDiff-B in the paper, set the model hyperparameters as:
~~~
--image_size 256 --num_channels 128 --class_cond False --num_res_blocks 2 --num_heads 1 --learn_sigma True --use_scale_shift_norm False --attention_resolutions 16 
~~~
diffusion hyperparameters as:
~~~
--diffusion_steps 1000 --noise_schedule linear --rescale_learned_sigmas False --rescale_timesteps False
~~~
To speed up the sampling:
~~~
--diffusion_steps 50 --dpm_solver True 
~~~
run on multiple GPUs:
~~~
--multi-gpu 0,1,2 (for example)
~~~
training hyperparameters as:
~~~
--lr 5e-5 --batch_size 8
~~~
and set ``--num_ensemble 5`` in sampling.
