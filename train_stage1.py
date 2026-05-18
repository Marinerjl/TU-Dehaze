import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from pathlib import Path
import commentjson as json

from models.Haze_transfer import Haze_transferNet
from models.Discriminator import Discriminator

from datasets import UnpairedLoader

from loss.loss import GANLoss, ContrastLoss

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
warnings.simplefilter('ignore', Warning, lineno=0)
torch.set_default_dtype(torch.float32)

def main():
    with open('./configs/Haze_Transfer.json', 'r') as f:
        args = json.load(f)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu_id"]
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    print('start')

    netG = torch.nn.DataParallel(Haze_transferNet()).cuda()

    netP_global = torch.nn.DataParallel(Discriminator()).cuda()

    net = {'G': netG, 'P_global': netP_global}

    optimizerG = optim.Adam(netG.parameters(), lr=args['lr'])
    optimizerP_global = optim.Adam(netP_global.parameters(), lr=args['lr'])

    optimizer = {'G': optimizerG, 'P_global': optimizerP_global}

    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(str(Path(args['resume'])), map_location='cpu')
            args['epoch_start'] = checkpoint['epoch']
            optimizerG.load_state_dict(checkpoint['optimizer_state_dict']['G'])
            netG.load_state_dict(checkpoint['model_state_dict']['G'])
            print('=> Loaded checkpoint {:s} (epoch {:d})'.format(args['resume'], checkpoint['epoch']))
        else:
            sys.exit('Please provide corrected model path!')
    else:
        args['epoch_start'] = 0
        if not Path(args['log_dir']).is_dir():
            Path(args['log_dir']).mkdir()
        if not Path(args['model_dir']).is_dir():
            Path(args['model_dir']).mkdir()

    for key, value in args.items():
        print('{:<15s}: {:s}'.format(key, str(value)))
    datasets = UnpairedLoader(root_dir="./data/train/unpaired/")

    print('\nBegin training with GPU: ' + (args['gpu_id']))
    train_epoch(net, datasets, optimizer, args)

def train_epoch(net, datasets, optimizer, args):
    criterion = nn.L1Loss().cuda()
    mse = nn.MSELoss().cuda()
    criterionGAN = GANLoss(args['gan_mode']).cuda()
    contrastive_loss = ContrastLoss().cuda()
    batch_size = args['batch_size']
    data_unpaire = DataLoader(datasets, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    num_iter_epoch = len(data_unpaire)

    for epoch in range(args['epoch_start'], args['epochs']):

        loss_epoch = {x: 0 for x in ['GL', 'PL']}

        tic = time.time()

        net['G'].train()
        net['P_global'].train()

        # 调整学习率
        current_lr = adjust_learning_rate(epoch, args)
        optimizer['G'].param_groups[0]['lr'] = current_lr
        optimizer['P_global'].param_groups[0]['lr'] = current_lr
        # optimizer['P_local2'].param_groups[0]['lr'] = current_lr
        # optimizer['P_local4'].param_groups[0]['lr'] = current_lr

        lr_G = optimizer['G'].param_groups[0]['lr']

        if lr_G < 1e-6:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        # output_dir = "./data/a/"
        for ii, data in enumerate(data_unpaire):

            hazy_img = data['hazy']
            clear_img = data['clear']
            depth_img = data['depth']
            # clear_name = data['clearname']

            hazy_img = hazy_img.to(device)
            clear_img = clear_img.to(device)
            depth_img = depth_img.to(device)

            optimizer['G'].zero_grad()

            set_requires_grad([net['P_global']], False)

            fake_hazy = net['G'](hazy_img, clear_img, depth_img)

            result = net['P_global'](fake_hazy)

            adversarial_loss1 = criterionGAN(result, True)

            adversarial_loss = adversarial_loss1

            loss_G = adversarial_loss

            loss_epoch['GL'] += loss_G.item()

            loss_G.backward(retain_graph=True)

            optimizer['G'].step()

            if (ii+1) % args['num_critic'] == 0:
                set_requires_grad([net['P_global']], True)

                pred_real_global = net['P_global'](hazy_img)
                pred_fake_global = net['P_global'](fake_hazy.detach())

                loss_P_real_global = criterionGAN(pred_real_global, True)
                loss_P_fake_global = criterionGAN(pred_fake_global, False)


                loss_P_global = (loss_P_real_global + loss_P_fake_global) * 0.5

                loss_P = loss_P_global

                loss_P.backward(retain_graph=True)

                optimizer['P_global'].step()

                optimizer['P_global'].zero_grad()

                loss_epoch['PL'] += loss_P.item()

                if (ii + 1) % args['print_freq'] == 0:
                    template = '[Epoch:{:>2d}/{:<3d}] {:s}:{:0>5d}/{:0>5d},' + \
                                   'PL:{:>6.6f},GL:{:>6.6f}'

                    print(template.format(epoch + 1, args['epochs'], phase, ii + 1, num_iter_epoch,
                                              loss_P.item(), loss_G.item()))

        loss_epoch['GL'] /= (ii + 1)
        loss_epoch['PL'] /= (ii + 1)


        print('Epoch [{0}]\t''lr_G: {lr:.6f}\t''Loss_P: {loss_P:.5f}' 'Loss_G: {loss_G:.5f}'.format(epoch, lr=lr_G, loss_P=loss_epoch['PL'], loss_G=loss_epoch['GL']))

        print('-' * 150)


        # save model
        model_prefix = 'model_'

        save_path_model = str(Path(args['model_dir']) / (model_prefix + str(epoch + 1))) + '.pth'
        torch.save({
                'epoch': epoch + 1,
                'model_state_dict': {x: net[x].state_dict() for x in ['G']},
                'optimizer_state_dict': {x: optimizer[x].state_dict() for x in ['G']},
        }, save_path_model)
        toc = time.time()
        print('This epoch take time {:.2f}'.format(toc - tic))

    print('Reach the maximal epochs! Finish training')

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or no
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def adjust_learning_rate(epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt['lr'] * (opt['gamma'] ** ((epoch) // opt['lr_decay']))
    # lr = opt['lr']
    return lr

if __name__ == '__main__':
    main()
