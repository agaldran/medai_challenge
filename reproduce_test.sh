#!/usr/bin/env bash

# GENERATE SEGMENTATIONS:#
# you first need to download test data from https://drive.google.com/drive/folders/1t8B45D2p3zEePHhUH5Qe-3iLs4EIrPJI
# then place it inside the data folder and run the following to generate both images and uncertainty maps
## polyps:
python generate_segs_with_uncertainty.py --save_path polyps/fpnet_resnext101_W_SAMADAM_ENS
# instruments:
python generate_segs_with_uncertainty.py --im_folder data/MedAI_2021_Instrument_Segmentation_Test_Dataset --load_checkpoint instruments/fpnet_resnext101_W_SAMADAM_F --save_path instruments/fpnet_resnext101_W_SAMADAM_ENS

# The following commands will generate segmentations that are overlayed on top of the test images:
## polyps:
python generate_overlay_segs.py --save_path polyps/fpnet_resnext101_W_SAMADAM_ENS
# instruments:
python generate_overlay_segs.py --im_folder data/MedAI_2021_Instrument_Segmentation_Test_Dataset --load_checkpoint instruments/fpnet_resnext101_W_SAMADAM_F --save_path instruments/fpnet_resnext101_W_SAMADAM_ENS
