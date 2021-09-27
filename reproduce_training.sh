#!/usr/bin/env bash
# POLYP EXPERIMENTS:
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/polyps/train_f1.csv --save_path polyps/fpnet_resnext101_W_SAMADAM_F1 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/polyps/train_f2.csv --save_path polyps/fpnet_resnext101_W_SAMADAM_F2 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/polyps/train_f3.csv --save_path polyps/fpnet_resnext101_W_SAMADAM_F3 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/polyps/train_f4.csv --save_path polyps/fpnet_resnext101_W_SAMADAM_F4 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/polyps/train_f5.csv --save_path polyps/fpnet_resnext101_W_SAMADAM_F5 --patience 2 --optimizer adam --max_lr 1e-4
# INSTRUMENT EXPERIMENTS:
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/instruments/train_f1.csv --save_path instruments/fpnet_resnext101_W_SAMADAM_F1 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/instruments/train_f2.csv --save_path instruments/fpnet_resnext101_W_SAMADAM_F2 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/instruments/train_f3.csv --save_path instruments/fpnet_resnext101_W_SAMADAM_F3 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/instruments/train_f4.csv --save_path instruments/fpnet_resnext101_W_SAMADAM_F4 --patience 2 --optimizer adam --max_lr 1e-4
python train_sam_patience.py --cycle_lens 25/10 --model_name fpnet_resnext101_W --csv_train data/instruments/train_f5.csv --save_path instruments/fpnet_resnext101_W_SAMADAM_F5 --patience 2 --optimizer adam --max_lr 1e-4

# GENERATE SEGMENTATIONS:#
# you first need to download test data from https://drive.google.com/drive/folders/1t8B45D2p3zEePHhUH5Qe-3iLs4EIrPJI
# then place it inside the data folder and run the following to generate both images and uncertainty maps
## polyps:
#python generate_segs_with_uncertainty.py --save_path polyps/fpnet_resnext101_W_SAMADAM_ENS
# instruments:
#python generate_segs_with_uncertainty.py --im_folder data/MedAI_2021_Instrument_Segmentation_Test_Dataset --load_checkpoint instruments/fpnet_resnext101_W_SAMADAM_F --save_path instruments/fpnet_resnext101_W_SAMADAM_ENS

# The following commands will generate segmentations that are overlayed on top of the test images:
## polyps:
#python generate_overlay_segs.py --save_path polyps/fpnet_resnext101_W_SAMADAM_ENS
# instruments:
#python generate_overlay_segs.py --im_folder data/MedAI_2021_Instrument_Segmentation_Test_Dataset --load_checkpoint instruments/fpnet_resnext101_W_SAMADAM_F --save_path instruments/fpnet_resnext101_W_SAMADAM_ENS
