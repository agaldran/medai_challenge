### Summary:
This repository contains all the necessary code to reproduce our results for the [MedAI](https://www.nora.ai/Competition/image-segmentation.html) competition on polyp and surgical instrument segmentation from endoscopic images. All the details of our model can be found in this paper:
```
Double Encoder-Decoder Networks for Gastrointestinal Polyp Segmentation
Adrian Galdran, Gustavo Carneiro, Miguel Ángel González Ballester
ICPR Artificial Intelligence for Healthcare Applications Workshop (AIHA), 2021

```
of which you will find a copy in this repo, within the `assets` folder. You can also watch me give some explanations on why our approach to this problem works pretty decently, the video is also in the `assets` folder. Have fun!

### Reproducibility: 
Install the provided conda environment by running:
```
conda create --name medai --file environment.txt
conda activate medai
```
Then you want to download the training data and get it ready, which can be done by running:
```
sh get_data.sh
```
Now, I trained our submissions with a 5-fold cross-validation approach, so if you want to train all from scratch it will take you, well, several days. The commands to carry out that training can be executed sequentially by running:
```
sh reproduce_training.sh
```

I will try to upload my pretrained weights when time allows, but my poor internet connection (I am in the middle of a long trip) makes it complicated for now, please be patient.

Once the models have been trained, you can also generate predicitions and uncertainty maps by running:
```
sh reproduce_test.sh
```
But you will first need to download test data from [this link](https://drive.google.com/drive/folders/1t8B45D2p3zEePHhUH5Qe-3iLs4EIrPJI), then uncompress it and place it inside the `data` folder.

