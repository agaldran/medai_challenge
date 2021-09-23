import sys, json, os, argparse
import warnings
from skimage.io import imsave
from skimage import img_as_ubyte
import os.path as osp
import pandas as pd
from tqdm import tqdm
from tqdm import trange
import numpy as np
import torch
from models.get_model import get_arch
from models.get_model import SMP_W
from utils.get_loaders import get_inference_seg_loader

from utils.model_saving_loading import str2bool, load_model
from utils.reproducibility import set_seeds
from skimage.filters import threshold_otsu as threshold
from scipy.ndimage import binary_fill_holes as bfh
from skimage.transform import resize

# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:

parser.add_argument('--im_folder', type=str, default='data/EndoTect_2020_Segmentation_Test_Dataset/images/', help='path to test data images') # or val4.csv
parser.add_argument('--model_name', type=str, default='fpnet_mobilenet_W', help='architecture')
parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
parser.add_argument('--im_size', help='delimited list input, could be 512, or 512,640', type=str, default='512,640')
parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=True, help='test time augmentation')
parser.add_argument('--load_checkpoint', type=str, default='experiments/polyps/fpnet_mobilenet_W_F1/', help='path to pre-trained checkpoint')
parser.add_argument('--save_path', type=str, default='results/polyps/fpnet_mobilenet_W_F1/', help='path to save predictions')


def generate_preds(loader, model, save_results_path, tta=True):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.eval()
    act = torch.sigmoid

    with trange(len(loader)) as t:
        for i_batch, (inputs, im_names, orig_sizes) in enumerate(loader):
            inputs= inputs.to(device)
            logits_seg = model(inputs)
            probs = act(logits_seg)
            if tta:
                logits_seg_lr = model(inputs.flip(-1))
                logits_seg_lr = logits_seg_lr.flip(-1)

                logits_seg_ud = model(inputs.flip(-2))
                logits_seg_ud = logits_seg_ud.flip(-2)

                logits_seg_lr_ud= model(inputs.flip(-1).flip(-2))
                logits_seg_lr_ud = logits_seg_lr_ud.flip(-1).flip(-2)

                probs_lr = act(logits_seg_lr)
                probs_ud = act(logits_seg_ud)
                probs_lr_ud = act(logits_seg_lr_ud)

                probs = torch.mean(torch.stack([probs, probs_lr, probs_ud, probs_lr_ud]), dim=0)
            for j in range(len(logits_seg)):
                segmentation = probs[j].detach().cpu().numpy()[0]

                try: thresh = threshold(segmentation)
                except: thresh = 0.5
                segmentation_bin = segmentation > thresh

                im_size = orig_sizes[1][j].item(), orig_sizes[0][j].item()
                im_name = im_names[j].split('/')[-1]
                im_name_out_seg = im_name.split('.')[-2] + '.png'

                seg_bin_resized = resize(segmentation_bin, output_shape=im_size, order=0)
                seg_bin_resized = bfh(seg_bin_resized )

                # save grayscale predictions
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    imsave(osp.join(save_results_path, im_name_out_seg),  img_as_ubyte((255 * seg_bin_resized).astype(np.uint8)))
            t.update()

if __name__ == '__main__':
    '''
    Example:
    python generate_segs.py

    '''
    data_path = 'data'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name
    tta = args.tta
    bs = args.batch_size
    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size)==1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size)==2:
        tg_size = (im_size[0], im_size[1])
    else: sys.exit('im_size should be a number or a tuple of two numbers')

    save_results_path = osp.join(args.save_path, 'segs')
    os.makedirs(save_results_path, exist_ok=True)
    load_checkpoint = osp.join('experiments', args.load_checkpoint)

    n_classes = 1
    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=n_classes, pretrained=False)

    if isinstance(model, SMP_W):
        model.mode = 'eval' # so that we do not return intermediate logits in a W-Net

    print('* Loading weights from previous checkpoint={}'.format(load_checkpoint))
    model, _ = load_model(model, load_checkpoint, device='cpu', with_opt=False)
    model = model.to(device)

    # generate test predictions
    test_loader = get_inference_seg_loader(args.im_folder, batch_size=bs, mean=mean, std=std, tg_size=tg_size)

    with torch.no_grad():
        print('Generating test predictions')
        generate_preds(test_loader, model, save_results_path, tta)
