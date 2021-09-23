import os, os.path as osp
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from PIL import Image

path_ims = 'data/polyps/images/'
path_masks = 'data/polyps/masks/'
im_list = os.listdir(path_ims)
mask_list = os.listdir(path_masks)

im_list = sorted([osp.join(path_ims, n) for n in im_list])
mask_list = sorted([osp.join(path_masks, n) for n in mask_list])

path_ims_out = 'data/polyps/images_512_640/'
path_masks_out = 'data/polyps/masks_512_640/'

os.makedirs(path_ims_out, exist_ok=True)
os.makedirs(path_masks_out, exist_ok=True)

tg_size = 640,512
im_list_out, m_list_out = [],[]

sizes = []

for i in tqdm(range(len(im_list))):
    im_name = im_list[i]
    m_name = mask_list[i]

    im_name_out = osp.join(path_ims_out, im_name.split('/')[-1])
    m_name_out = osp.join(path_masks_out, m_name.split('/')[-1].replace('jpg', 'png'))


    img = Image.open(im_name)
    mask = Image.open(m_name)
    mask = Image.fromarray(255*(np.array(mask)>127).astype(np.uint8)) # they are jpg with more than two values
    
    sizes.append([img.size[1],img.size[0]])
    
    img = img.resize(tg_size, Image.BICUBIC)
    mask = mask.resize(tg_size, Image.NEAREST)

    im_list_out.append(im_name_out)
    m_list_out.append(m_name_out)

    img.save(im_name_out)
    mask.save(m_name_out)
    
        
data_tuples = list(zip(im_list_out, m_list_out))
df = pd.DataFrame(data_tuples, columns=['image_path','mask_path'])

df_other, df_val1 = train_test_split(df, test_size=200, random_state=0)
df_other, df_val2 = train_test_split(df_other, test_size=200, random_state=0)
df_other, df_val3 = train_test_split(df_other, test_size=200, random_state=0)
df_val4, df_val5 = train_test_split(df_other, test_size=200, random_state=0)

df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5])
df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5])
df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5])
df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5])
df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4])

df_train1.to_csv('data/polyps/train_f1.csv', index=None)
df_val1.to_csv('data/polyps/val_f1.csv', index=None)

df_train2.to_csv('data/polyps/train_f2.csv', index=None)
df_val2.to_csv('data/polyps/val_f2.csv', index=None)

df_train3.to_csv('data/polyps/train_f3.csv', index=None)
df_val3.to_csv('data/polyps/val_f3.csv', index=None)

df_train4.to_csv('data/polyps/train_f4.csv', index=None)
df_val4.to_csv('data/polyps/val_f4.csv', index=None)

df_train5.to_csv('data/polyps/train_f5.csv', index=None)
df_val5.to_csv('data/polyps/val_f5.csv', index=None)

path_ims = 'data/instruments/images/'
path_masks = 'data/instruments/masks/'
im_list = sorted([osp.join(path_ims, n) for n in im_list])
mask_list = sorted([osp.join(path_masks, n) for n in mask_list])

im_list = [osp.join(path_ims, n) for n in im_list]
mask_list = [osp.join(path_masks, n) for n in mask_list]

path_ims_out = 'data/instruments/images_512_640/'
path_masks_out = 'data/instruments/masks_512_640/'

os.makedirs(path_ims_out, exist_ok=True)
os.makedirs(path_masks_out, exist_ok=True)

tg_size = 640,512
im_list_out, m_list_out = [],[]

sizes = []

for i in tqdm(range(len(im_list))):
    im_name = im_list[i]
    m_name = mask_list[i]

    im_name_out = osp.join(path_ims_out, im_name.split('/')[-1])
    m_name_out = osp.join(path_masks_out, m_name.split('/')[-1])
    

    img = Image.open(im_name)
    mask = Image.open(m_name)
    
    sizes.append([img.size[1],img.size[0]])
    
    img = img.resize(tg_size, Image.BICUBIC)
    mask = mask.resize(tg_size, Image.NEAREST)

    im_list_out.append(im_name_out)
    m_list_out.append(m_name_out)

    img.save(im_name_out)
    mask.save(m_name_out)
    
        
data_tuples = list(zip(im_list_out, m_list_out))
df = pd.DataFrame(data_tuples, columns=['image_path','mask_path'])

df_other, df_val1 = train_test_split(df, test_size=118, random_state=0)
df_other, df_val2 = train_test_split(df_other, test_size=118, random_state=0)
df_other, df_val3 = train_test_split(df_other, test_size=118, random_state=0)
df_val4, df_val5 = train_test_split(df_other, test_size=118, random_state=0)

df_train1 = pd.concat([df_val2, df_val3, df_val4, df_val5])
df_train2 = pd.concat([df_val1, df_val3, df_val4, df_val5])
df_train3 = pd.concat([df_val1, df_val2, df_val4, df_val5])
df_train4 = pd.concat([df_val1, df_val2, df_val3, df_val5])
df_train5 = pd.concat([df_val1, df_val2, df_val3, df_val4])

df_train1.to_csv('data/instruments/train_f1.csv', index=None)
df_val1.to_csv('data/instruments/val_f1.csv', index=None)

df_train2.to_csv('data/instruments/train_f2.csv', index=None)
df_val2.to_csv('data/instruments/val_f2.csv', index=None)

df_train3.to_csv('data/instruments/train_f3.csv', index=None)
df_val3.to_csv('data/instruments/val_f3.csv', index=None)

df_train4.to_csv('data/instruments/train_f4.csv', index=None)
df_val4.to_csv('data/instruments/val_f4.csv', index=None)

df_train5.to_csv('data/instruments/train_f5.csv', index=None)
df_val5.to_csv('data/instruments/val_f5.csv', index=None)

df_train5.to_csv('data/instruments/train_f5.csv', index=None)
df_val5.to_csv('data/instruments/val_f5.csv', index=None)
