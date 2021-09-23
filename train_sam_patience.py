import sys, json, os, time, argparse
import os.path as osp
from datetime import datetime
import operator
from tqdm import trange
import numpy as np
import torch
from models.get_model import get_arch

from utils.get_loaders import get_train_val_seg_loaders
from utils.model_saving_loading import save_model, str2bool
from utils.reproducibility import set_seeds
from skimage.filters import threshold_otsu as threshold
from utils.evaluation import dice_score
from utils.sam import SAM

from torch.optim.lr_scheduler import CosineAnnealingLR

# argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--csv_train', type=str, default='data/polyps/train_f1.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='fpnet_mobilenet_W', help='architecture')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--min_lr', type=float, default=1e-8, help='min learning rate')
parser.add_argument('--max_lr', type=float, default=1e-4, help='max learning rate')
parser.add_argument('--cycle_lens', type=str, default='25/10', help='cycling config (nr cycles/cycle len)')
parser.add_argument('--optimizer', type=str, default='adam', help='optimizer choice')
parser.add_argument('--patience', type=int, default=3, help='batch size')
parser.add_argument('--metric', type=str, default='dice', help='which metric to use for monitoring progress (loss/dice)')
parser.add_argument('--im_size', help='delimited list input, could be 512, or 512,640', type=str, default='512,640')
parser.add_argument('--do_not_save', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')
parser.add_argument('--num_workers', type=int, default=8, help='number of parallel (multiprocessing) workers to launch '
                                                               'for data loading tasks (handled by pytorch) [default: %(default)s]')
parser.add_argument('--n_checkpoints', type=int, default=1, help='nr of best checkpoints to keep (defaults to 3)')


def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def disable_bn(model):
  for module in model.modules():
    if isinstance(module, torch.nn.BatchNorm2d):
      module.eval()

def enable_bn(model):
  model.train()

def run_one_epoch(loader, model, criterion, optimizer=None, scheduler=None, assess=False):
    device = 'cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()
    if assess:
        act = torch.sigmoid if n_classes == 1 else torch.nn.Softmax(dim=0)
        dices_otsu = []

    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):

            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            logits_aux, logits = logits

            if model.n_classes == 1:  # BCEWithLogitsLoss()/DiceLoss()
                loss_aux = criterion(logits_aux, labels.float())
                loss = loss_aux + criterion(logits.float(), labels.float())
            else:  # CrossEntropyLoss()
                loss_aux = criterion(logits_aux, labels)
                loss = loss_aux + criterion(logits, labels)

            if train:  # only in training mode
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # Compute model output, loss, and update again
                logits = model(inputs)
                logits_aux, logits = logits
                loss_aux = criterion(logits_aux, labels.float())
                loss = loss_aux + criterion(logits.float(), labels.float())

                # compute BN statistics only in the first pass
                disable_bn(model)
                loss.backward()
                enable_bn(model)

                optimizer.second_step(zero_grad=True)

                scheduler.step()


            if assess:
                # evaluation
                for i in range(len(logits)):
                    prediction = act(logits[i]).detach().cpu().numpy()[-1]
                    target = labels[i].cpu().numpy()
                    try:
                        thresh = threshold(prediction)
                    except:
                        thresh = 0.5
                    bin_pred = prediction > thresh
                    dice = dice_score(target.ravel(), bin_pred.ravel())
                    dices_otsu.append(dice)

            # Compute running loss
            running_loss += loss.item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train:
                t.set_postfix(tr_loss_lr="{:.4f}/{:.6f}".format(float(run_loss), get_lr(optimizer)))
            else:
                t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()

    if assess: return np.mean(dices_otsu), np.std(dices_otsu), run_loss
    return None, None, None


def train_one_cycle(train_loader, model, criterion, optimizer=None, scheduler=None, cycle=0):
    model.train()
    optimizer.zero_grad()
    cycle_len = scheduler.cycle_lens[cycle]
    for epoch in range(cycle_len):
        print('Cycle {:d} | Epoch {:d}/{:d}'.format(cycle + 1, epoch + 1, cycle_len))
        if epoch == cycle_len - 1:
            assess = True  # only get logits/labels on last cycle
        else:
            assess = False
        tr_mean_dice, tr_std_dice, tr_loss = \
            run_one_epoch(train_loader, model, criterion, optimizer=optimizer,
                          scheduler=scheduler, assess=assess)

    return tr_mean_dice, tr_std_dice, tr_loss


def train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, metric, exp_path,
                n_checkpoints, patience):
    n_cycles = len(scheduler.cycle_lens)
    best_dice, best_cycle, best_models = 0, 0, []
    all_tr_losses, all_vl_losses = [], []
    all_tr_dices, all_vl_dices = [], []
    is_better, best_monitoring_metric = compare_op(metric)

    for cycle in range(n_cycles):
        print('Cycle {:d}/{:d}'.format(cycle + 1, n_cycles))
        # prepare next cycle:
        # reset iteration counter
        scheduler.last_epoch = -1
        # update number of iterations
        scheduler.T_max = scheduler.cycle_lens[cycle] * len(train_loader)

        # train one cycle
        tr_mean_dice, tr_std_dice, tr_loss = train_one_cycle(train_loader, model, criterion, optimizer, scheduler, cycle)

        with torch.no_grad():
            vl_mean_dice, vl_std_dice, vl_loss = run_one_epoch(val_loader, model, criterion, assess=True)

        all_tr_dices.append(tr_mean_dice)
        all_vl_dices.append(vl_mean_dice)
        all_tr_losses.append(tr_loss)
        all_vl_losses.append(vl_loss)

        print('Train||Val Loss: {:.4f}||{:.4f}  -- Train||Val DICE: {:.2f}+-{:.2f}||{:.2f} +- {:.2f}'.format(
            tr_loss, vl_loss, 100 * tr_mean_dice, 100 * tr_std_dice, 100 * vl_mean_dice, 100 * vl_std_dice))
        # check if performance was better than anyone before and checkpoint if so
        if metric == 'loss':
            monitoring_metric = vl_loss
        elif metric == 'dice':
            monitoring_metric = vl_mean_dice

        if n_checkpoints == 1:  # only save best val model
            if is_better(monitoring_metric, best_monitoring_metric):
                print('Best {} attained. {:.2f} --> {:.2f}'.format(metric, 100 * best_monitoring_metric,
                                                                   100 * monitoring_metric))
                best_loss, best_dice, best_cycle = vl_loss, vl_mean_dice, cycle
                best_monitoring_metric = monitoring_metric
                if exp_path is not None:
                    print(15 * '-', ' Checkpointing ', 15 * '-')
                    save_model(exp_path, model, optimizer)
            else:
                print('Best {} so far {:.2f} at cycle {:d}, {:} cycles to early stop.'.format(metric, 100 * best_monitoring_metric,
                                                                                        best_cycle+1, patience - cycle + best_cycle))
                if cycle-best_cycle == patience:
                    return best_dice, best_cycle+1, all_tr_dices, all_vl_dices, all_tr_losses, all_vl_losses

        else:  # SAVE n best - keep deleting worse ones
            from operator import itemgetter
            import shutil
            if exp_path is not None:
                s_name = 'cycle_{}_DICE_{:.2f}'.format(str(cycle + 1).zfill(2), 100 * vl_mean_dice)
                best_models.append([osp.join(exp_path, s_name), vl_mean_dice])

                if cycle < n_checkpoints:  # first n_checkpoints cycles save always
                    print('-------- Checkpointing to {}/ --------'.format(s_name))
                    save_model(osp.join(exp_path, s_name), model, optimizer)
                else:
                    worst_model = sorted(best_models, key=itemgetter(1), reverse=True)[-1][0]
                    if s_name != worst_model:  # this model was better than one of the best n_checkpoints models, remove that one
                        print('-------- Checkpointing to {}/ --------'.format(s_name))
                        save_model(osp.join(exp_path, s_name), model, optimizer)
                        print('----------- Deleting {}/ -----------'.format(worst_model.split('/')[-1]))
                        shutil.rmtree(worst_model)
                        best_models = sorted(best_models, key=itemgetter(1), reverse=True)[:n_checkpoints]

    del model
    torch.cuda.empty_cache()
    return best_dice, best_cycle+1, all_tr_dices, all_vl_dices, all_tr_losses, all_vl_losses


if __name__ == '__main__':

    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    n_classes = 1

    # gather parser parameters
    model_name = args.model_name
    optimizer_choice = args.optimizer
    patience = args.patience
    max_lr, min_lr, bs= args.max_lr, args.min_lr, args.batch_size

    cycle_lens, metric = args.cycle_lens.split('/'), args.metric
    cycle_lens = list(map(int, cycle_lens))

    if len(cycle_lens) == 2:  # handles option of specifying cycles as pair (n_cycles, cycle_len)
        cycle_lens = cycle_lens[0] * [cycle_lens[1]]

    im_size = tuple([int(item) for item in args.im_size.split(',')])
    if isinstance(im_size, tuple) and len(im_size) == 1:
        tg_size = (im_size[0], im_size[0])
    elif isinstance(im_size, tuple) and len(im_size) == 2:
        tg_size = (im_size[0], im_size[1])
    else:
        sys.exit('im_size should be a number or a tuple of two numbers')

    do_not_save = str2bool(args.do_not_save)
    if do_not_save is False:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path = osp.join('experiments', save_path)
        args.experiment_path = experiment_path
        os.makedirs(experiment_path, exist_ok=True)

        config_file_path = osp.join(experiment_path, 'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else:
        experiment_path = None
    n_checkpoints = args.n_checkpoints

    csv_train = args.csv_train
    csv_val = csv_train.replace('train', 'val')

    print('* Instantiating a {} model'.format(model_name))
    model, mean, std = get_arch(model_name, n_classes=n_classes)
    print('* Creating Dataloaders, batch size = {}, workers = {}'.format(bs, args.num_workers))
    train_loader, val_loader = get_train_val_seg_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs,
                                                         tg_size=tg_size, mean=mean, std=std,
                                                         num_workers=args.num_workers)

    model = model.to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    if optimizer_choice == 'adam':
        base_optimizer = torch.optim.Adam
    elif optimizer_choice == 'sgd':
        base_optimizer = torch.optim.SGD
    else:
        sys.exit('please choose between adam or sgd optimizers')

    optimizer = SAM(model.parameters(), base_optimizer, lr=max_lr)

    scheduler = CosineAnnealingLR(optimizer.base_optimizer, T_max=cycle_lens[0] * len(train_loader), eta_min=min_lr)

    setattr(optimizer, 'max_lr', max_lr)  # store it inside the optimizer for accessing to it later
    setattr(scheduler, 'cycle_lens', cycle_lens)

    criterion = torch.nn.BCEWithLogitsLoss() if model.n_classes == 1 else torch.nn.CrossEntropyLoss()

    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n', '-' * 10)
    start = time.time()
    m1, m2, all_tr_dices, all_vl_dices, all_tr_losses, all_vl_losses = \
        train_model(model, optimizer, criterion, train_loader, val_loader, scheduler, metric, experiment_path, n_checkpoints, patience)
    end = time.time()

    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    print('Training time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds))
    print('Best Dice: %f' % m1)
    print('Best Cycle: %d' % m2)
    if do_not_save is False:
        with open(osp.join(experiment_path, 'val_metrics.txt'), 'w') as f:
            print('Best DICE = {:.2f}\nBest cycle = {}'.format(100 * m1, m2), file=f)
            for j in range(len(all_tr_dices)):
                print('Cycle = {} -> DICE={:.2f}/{:.2f}, Loss={:.4f}/{:.4f} '.format(j + 1,
                                     100*all_tr_dices[j], 100*all_vl_dices[j],
                                     100 * all_tr_losses[j], 100 * all_vl_losses[j]), file=f)
            print('\nTraining time: {:0>2}h {:0>2}min {:05.2f}secs'.format(int(hours), int(minutes), seconds), file=f)