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
from scipy.ndimage import binary_fill_holes as bfh
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from skimage import filters

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--im_folder', type=str, default='data/MedAI_2021_Polyp_Segmentation_Test_Dataset', help='path to test data images')
# or 'data/MedAI_2021_Instrument_Segmentation_Test_Dataset'
parser.add_argument('--model_name', type=str, default='fpnet_resnext101_W', help='architecture')
parser.add_argument('--temp', type=float, default=1, help='temperature for mixing each fold model')
parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
parser.add_argument('--im_size', help='delimited list input, could be 512, or 512,640', type=str, default='512,640')
parser.add_argument('--tta', type=str2bool, nargs='?', const=True, default=True, help='test time augmentation')
parser.add_argument('--load_checkpoint', type=str, default='polyps/fpnet_resnext101_W_SAMADAM_F', help='path to pre-trained checkpoint')
# or 'instruments/fpnet_resnext101_W_SAMADAM_F'
parser.add_argument('--save_path', type=str, default='polyps/fpnet_resnext101_W_SAMADAM_ENS', help='path to save predictions')
# or 'instruments/fpnet_resnext101_W_SAMADAM_ENS'
parser.add_argument('--save_path_unc', type=str, default='results_uncertainty.csv', help='path to save predictions')

def mutual_information(im1,im2):
    """
    Mutual information for joint histogram
    https://matthew-brett.github.io/teaching/mutual_information.html
    """
    # Convert bins counts to probability values
    hist_2d, _, _= np.histogram2d(im1.ravel(), im2.ravel(), bins = 20)
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def generate_preds(loader, model, states, temp, save_results_path, tta, device, save_path_unc):
    os.makedirs(save_results_path, exist_ok=True)
    model.to(device)
    tq_loader = tqdm(enumerate(loader), total=len(loader))

    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    model.eval()
    uncertainties, im_names_ = [], []

    for i_batch, (inputs, im_names, orig_sizes) in tq_loader:
        inputs = inputs.to(device)
        avg_probs_intermediate = []
        avg_probs_final = []
        for state in states:
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            logits_seg = model(inputs)
            probs_intermediate, probs_final = logits_seg[0].sigmoid(), logits_seg[1].sigmoid()

            if tta:
                logits_seg_lr_intermediate, logits_seg_lr_final = model(inputs.flip(-1))
                logits_seg_lr_intermediate, logits_seg_lr_final = logits_seg_lr_intermediate.flip(-1), logits_seg_lr_final.flip(-1)

                logits_seg_ud_intermediate, logits_seg_ud_final = model(inputs.flip(-2))
                logits_seg_ud_intermediate, logits_seg_ud_final = logits_seg_ud_intermediate.flip(-2), logits_seg_ud_final.flip(-2)

                logits_seg_lr_ud_intermediate, logits_seg_lr_ud_final = model(inputs.flip(-1).flip(-2))
                logits_seg_lr_ud_intermediate, logits_seg_lr_ud_final = logits_seg_lr_ud_intermediate.flip(-1).flip(-2), logits_seg_lr_ud_final.flip(-1).flip(-2)

                probs_lr_intermediate, probs_lr_final = logits_seg_lr_intermediate.sigmoid(), logits_seg_lr_final.sigmoid()
                probs_ud_intermediate, probs_ud_final = logits_seg_ud_intermediate.sigmoid(), logits_seg_ud_final.sigmoid()
                probs_lr_ud_intermediate, probs_lr_ud_final = logits_seg_lr_ud_intermediate.sigmoid(), logits_seg_lr_ud_final.sigmoid()

                probs_intermediate = torch.mean(torch.stack([probs_intermediate, probs_lr_intermediate,
                                                             probs_ud_intermediate, probs_lr_ud_intermediate]), dim=0)
                probs_final = torch.mean(torch.stack([probs_final, probs_lr_final, probs_ud_final, probs_lr_ud_final]), dim=0)

            avg_probs_intermediate.append(probs_intermediate)
            avg_probs_final.append(probs_final)

        probs_intermediate = 0.20*(avg_probs_intermediate[0]**temp + avg_probs_intermediate[1]**temp +
                                     avg_probs_intermediate[2]**temp + avg_probs_intermediate[3]**temp + avg_probs_intermediate[4]**temp).cpu()
        probs_final = 0.20*(avg_probs_final[0] ** temp + avg_probs_final[1] ** temp + avg_probs_final[2] ** temp
                              + avg_probs_final[3] ** temp + avg_probs_final[4] ** temp).cpu()

        for j in range(len(probs_final)):
            segmentation = probs_final[j].detach().cpu().numpy()[0]
            pre_segmentation = probs_intermediate[j].detach().cpu().numpy()[0]

            uncertainty_map = np.abs(segmentation - pre_segmentation)
            unc_score_norm = mutual_information(segmentation, pre_segmentation)


            thresh = 0.60
            segmentation_bin = segmentation > thresh

            im_size = orig_sizes[1][j].item(), orig_sizes[0][j].item()
            im_name = im_names[j].split('/')[-1]
            im_name_out_seg = im_name.split('.')[-2] + '.png'

            seg_bin_resized = resize(segmentation_bin, output_shape=im_size, order=0)
            seg_bin_resized = bfh(seg_bin_resized)
            uncertainty_resized = resize(uncertainty_map, output_shape=im_size, order=1)

            im_names_.append(im_name)
            uncertainties.append(100 * unc_score_norm)

            # save grayscale predictions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                imsave(osp.join(save_results_path, im_name_out_seg),
                       img_as_ubyte((255 * seg_bin_resized).astype(np.uint8)))
                imsave(osp.join(save_path_unc, im_name_out_seg.replace('.png', '.jpg')),
                       img_as_ubyte((255 * uncertainty_resized).astype(np.uint8)))

    data_tuples = list(zip(im_names_, uncertainties))
    return pd.DataFrame(data_tuples, columns=['im_name','mutual_info'])


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

    save_results_path = osp.join('results', args.save_path, 'segs')
    save_path_unc = osp.join('results', args.save_path, 'unc_maps')
    os.makedirs(save_results_path, exist_ok=True)
    os.makedirs(save_path_unc, exist_ok=True)

    checkpoint_list = [osp.join('experiments', args.load_checkpoint + str(i), 'model_checkpoint.pth') for i in [1, 2, 3, 4, 5]]
    states = [torch.load(checkpoint_list[i], map_location=device) for i in [0, 1, 2, 3, 4]]

    n_classes = 1
    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=n_classes, pretrained=False)

    # if isinstance(model, SMP_W):
    #     model.mode = 'eval' # so that we do not return intermediate logits in a W-Net

    # generate test predictions
    test_loader = get_inference_seg_loader(args.im_folder, batch_size=bs, mean=mean, std=std, tg_size=tg_size)
    with torch.no_grad():
        print('Generating test predictions')
        df_unc = generate_preds(test_loader, model, states, args.temp, save_results_path, tta, device, save_path_unc)

    df_unc.sort_values('mutual_info').to_csv(osp.join('results', args.save_path, 'sorted_mi.csv'), index=None)
