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
parser.add_argument('--model_name', type=str, default='fpnet_resnext101_W', help='architecture')
parser.add_argument('--temp', type=float, default=1, help='temperature for mixing each fold model')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--im_size', help='delimited list input, could be 512, or 512,640', type=str, default='512,640')
parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=True, help='test time augmentation')
parser.add_argument('--load_checkpoint', type=str, default='polyps/fpnet_resnext101_W_SAMADAM1_F', help='path to pre-trained checkpoints')
parser.add_argument('--leave_out', type=int, default=5, help='fold to leave out in ensemble')
parser.add_argument('--save_path', type=str, default='results/polyps/fpnet_resnext101_W_SAMADAM_ENS1234/', help='path to save predictions')


def generate_preds(loader, model, states, temp, save_results_path, tta, device):
    os.makedirs(save_results_path, exist_ok=True)
    model.to(device)
    tq_loader = tqdm(enumerate(loader), total=len(loader))

    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.eval()

    for i_batch, (inputs, im_names, orig_sizes) in tq_loader:
        inputs = inputs.to(device)
        avg_probs = []
        for state in states:
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            logits_seg = model(inputs)
            probs_seg = logits_seg.sigmoid()

            if tta:
                logits_seg_lr = model(inputs.flip(-1)).flip(-1)
                probs_seg_lr = logits_seg_lr.sigmoid()
                logits_seg_ud = model(inputs.flip(-2)).flip(-2)
                probs_seg_ud = logits_seg_ud.sigmoid()
                logits_seg_lr_ud = model(inputs.flip(-1).flip(-2)).flip(-1).flip(-2)
                probs_seg_lr_ud = logits_seg_lr_ud.sigmoid()
                probs_seg = 0.25 * (probs_seg + probs_seg_lr + probs_seg_ud + probs_seg_lr_ud)
            avg_probs.append(probs_seg)

        probs = 0.25 * (avg_probs[0]**temp + avg_probs[1]**temp + avg_probs[2]**temp + avg_probs[3]**temp).cpu()

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
                imsave(osp.join(save_results_path, im_name_out_seg),  (255 * seg_bin_resized).astype(np.uint8))


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
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else: sys.exit('im_size should be a number or a tuple of two numbers')

    save_results_path = osp.join(args.save_path, 'segs')
    os.makedirs(save_results_path, exist_ok=True)
    print('Reading models from {}'.format(args.load_checkpoint))
    leave_out = args.leave_out
    c_list = [1, 2, 3, 4, 5]
    c_list.remove(leave_out)
    checkpoint_list = [osp.join('experiments', args.load_checkpoint + str(i), 'model_checkpoint.pth') for i in c_list]
    states = [torch.load(c, map_location=device) for c in checkpoint_list]

    n_classes = 1
    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=n_classes, pretrained=False)
    model.mode = 'eval' # so that we do not return intermediate logits in a W-Net

    # generate test predictions
    test_loader = get_inference_seg_loader(args.im_folder, batch_size=bs, mean=mean, std=std, tg_size=tg_size)

    with torch.no_grad():
        print('Generating test predictions')
        generate_preds(test_loader, model, states, args.temp, save_results_path, tta, device)
