import random
import numpy as np
import torch

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from models.MSBDN import MSBDNNet
from models.DCPDN import DCPDN
from models.AtmLocal import AtmLocal
from models.Haze_transfer import Haze_transferNet
from models.Discriminator import Discriminator
from torchvision.models import vgg16
from utils.metrics import SSIM
from datasets import Test_Syn_Loader, UnpairedLoader
from utils import *
from loss.loss import GANLoss
from loss.PerceptualLoss import LossNetwork as PerLoss
import warnings
from pathlib import Path
import commentjson as json


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# filter warnings
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')
warnings.simplefilter('ignore', Warning, lineno=0)
torch.set_default_dtype(torch.float32)


def main():
    with open('./configs/dehaze.json', 'r') as f:
        args = json.load(f)
    # 将CUDA_DEVICE_ORDER设置为'PCI_BUS_ID'可以确保设备根据其PCI总线顺序分配ID
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = args["gpu_id"]
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # build up the dehaze
    print('start')

    #  RNet
    netD = torch.nn.DataParallel(MSBDNNet()).cuda()

    #  SNet
    netT = torch.nn.DataParallel(DCPDN()).cuda()

    #  LNet
    netA = torch.nn.DataParallel(AtmLocal(3, 3)).cuda()

    ckp = torch.load("")
    netG = torch.nn.DataParallel(Haze_transferNet()).cuda()
    netG.load_state_dict(ckp['model_state_dict']['G'])

    # build up the discriminator
    netP_c = torch.nn.DataParallel(Discriminator()).cuda()

    criterionGAN = GANLoss(args['gan_mode']).cuda()
    criterion = []
    criterion.append(nn.L1Loss().to(device))
    vgg_model = vgg16(pretrained=True).features[:16]
    vgg_model = vgg_model.to(device)
    for param in vgg_model.parameters():
        param.requires_grad = False
    criterion.append(PerLoss(vgg_model).to(device))

    init_weights(netD, init_type='normal', init_gain=0.02)
    init_weights(netA, init_type='normal', init_gain=0.02)
    init_weights(netT, init_type='normal', init_gain=0.02)
    init_weights(netP_c, init_type='normal', init_gain=0.02)

    net = {'D': netD, 'T': netT, 'A': netA, 'G': netG, 'P_c': netP_c}

    # optimizer
    # dehaze
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr_D'])
    optimizerT = optim.Adam(netT.parameters(), lr=args['lr_T'])
    optimizerA = optim.Adam(netA.parameters(), lr=args['lr_T'])
    # haze_transfernet_generator
    optimizerG = optim.Adam(netG.parameters(), lr=args['lr_G'])
    # haze_transfernet_discriminator
    optimizerP_c = optim.Adam(netP_c.parameters(), lr=args['lr_P'])

    optimizer = {'D': optimizerD, 'T': optimizerT, 'A': optimizerA, 'G': optimizerG, 'P_c': optimizerP_c}

    if args['resume']:
        if Path(args['resume']).is_file():
            print('=> Loading checkpoint {:s}'.format(str(Path(args['resume']))))
            checkpoint = torch.load(str(Path(args['resume'])), map_location='cpu')
            args['epoch_start'] = checkpoint['epoch']
            optimizerD.load_state_dict(checkpoint['optimizer_state_dict']['D'])
            optimizerT.load_state_dict(checkpoint['optimizer_state_dict']['T'])
            optimizerA.load_state_dict(checkpoint['optimizer_state_dict']['A'])
            optimizerP_c.load_state_dict(checkpoint['optimizer_state_dict']['P_c'])
            netD.load_state_dict(checkpoint['model_state_dict']['D'])
            netT.load_state_dict(checkpoint['model_state_dict']['T'])
            netA.load_state_dict(checkpoint['model_state_dict']['A'])
            netP_c.load_state_dict(checkpoint['model_state_dict']['P_c'])
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
    test_datasets = Test_Syn_Loader(root_dir="./data/test/HSTS/", size=256, edge_decay=0, data_augment=True,
                               cache_memory=False)
    # train model
    print('\nBegin training with GPU: ' + (args['gpu_id']))
    train_epoch(net, datasets, test_datasets, optimizer, args, criterionGAN, criterion)


def train_epoch(net, datasets, test_datasets, optimizer, args, criterionGAN, criterion):
    data_unpaire = DataLoader(datasets, batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    test_data_loader = DataLoader(test_datasets, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)

    num_iter_epoch = len(data_unpaire)
    num_iter_epoch_test = len(test_data_loader)
    # num_iter_epoch = ceil(num_data/ args['batch_size'])

    for epoch in range(args['epoch_start'], args['epochs']):
        loss_epoch = {x: 0 for x in ['PL_h', 'PL_c', 'DL', 'GL']}
        subloss_epoch = {x: 0 for x in
                         ['loss_recon', 'perceptual_loss', 'dc_loss', 'tv_loss', 'adv_loss', 'loss_P_real_h',
                          'loss_P_fake_h', 'loss_P_real_c', 'loss_P_fake_c']}

        tic = time.time()

        # train stage
        net['D'].train()
        net['T'].train()
        net['A'].train()
        net['G'].eval()
        net['P_c'].train()

        set_requires_grad([net['G']], False)

        current_lr = adjust_learning_rate(epoch, args)
        optimizer['D'].param_groups[0]['lr'] = current_lr
        optimizer['T'].param_groups[0]['lr'] = current_lr
        optimizer['A'].param_groups[0]['lr'] = current_lr
        optimizer['P_c'].param_groups[0]['lr'] = current_lr

        lr_D = optimizer['D'].param_groups[0]['lr']

        if lr_D < 1e-6:
            sys.exit('Reach the minimal learning rate')
        phase = 'train'
        for ii, data in enumerate(data_unpaire):

            hazy_img = data['hazy']
            clear_img = data['clear']
            depth_img = data['depth']

            hazy_img = hazy_img.to(device)
            clear_img = clear_img.to(device)
            depth_img = depth_img.to(device)

            #  training generator and dehaze
            optimizer['D'].zero_grad()
            optimizer['T'].zero_grad()
            optimizer['A'].zero_grad()

            # dehaze
            dehaze_img = net['D'](hazy_img)

            transmap = net['T'](hazy_img)

            airlight = net['A'](hazy_img)

            rec_real_haze = dehaze_img * transmap + (1 - transmap) * airlight

            fake_im_hazy1 = net['G'](hazy_img, clear_img, depth_img)

            dehaze_fake = net['D'](fake_im_hazy1)

            set_requires_grad([net['P_c']], False)

            adversarial_loss_c = criterionGAN(net['P_c'](dehaze_img), True)

            loss_recon1 = criterion[0](dehaze_fake, clear_img)

            loss_recon2 = criterion[0](rec_real_haze, hazy_img)

            loss_D = 10 * loss_recon1 + 10 * loss_recon2 + adversarial_loss_c * 0.5


            loss_D.backward(retain_graph=True)

            optimizer['D'].step()
            optimizer['T'].step()
            optimizer['A'].step()

            loss_epoch['DL'] += loss_D.item()


            #  training discriminator
            if (ii + 1) % args['num_critic'] == 0:
                set_requires_grad([net['P_c']], True)

                pred_real1 = net['P_c'](clear_img)
                loss_P_real_c = criterionGAN(pred_real1, True)
                pred_fake = net['P_c'](dehaze_img.detach())
                loss_P_fake_c = criterionGAN(pred_fake, False)

                # discriminator loss
                loss_P_c = (loss_P_real_c + loss_P_fake_c) * 0.5

                # Combined loss and calculate gradients
                loss_P_c.backward(retain_graph=True)

                optimizer['P_c'].step()
                optimizer['P_c'].zero_grad()

                loss_epoch['PL_c'] += loss_P_c.item()
                subloss_epoch['loss_P_real_c'] += loss_P_real_c.item()
                subloss_epoch['loss_P_fake_c'] += loss_P_fake_c.item()


                if (ii + 1) % args['print_freq'] == 0:
                    template = '[Epoch:{:>2d}/{:<3d}] {:s}:{:0>5d}/{:0>5d},' + \
                               'DL:{:>6.6f}, ' + \
                               'rec_loss1:{:>6.4f}, rec_loss2:{:>6.4f}, adv_loss:{:>6.4f}'
                    print(template.format(epoch + 1, args['epochs'], phase, ii + 1, num_iter_epoch,
                                    loss_D.item(),
                                    loss_recon1.item(), loss_recon2.item(), adversarial_loss_c.item()))


        loss_epoch['DL'] /= (ii + 1)
        subloss_epoch['loss_recon'] /= (ii + 1)

        loss_epoch['PL_c'] /= (ii + 1)
        subloss_epoch['loss_P_real_c'] /= (ii + 1)
        subloss_epoch['loss_P_fake_c'] /= (ii + 1)

        template = '{:s}: GL:{:>6.6f},DL:{:>6.6f}'


        print(template.format(phase, loss_epoch['PL_c'], loss_epoch['DL']))

        net['G'].eval()
        print('Epoch [{0}]\t'
              'lr: {lr:.6f}\t'
              'Loss: {loss:.5f}'.format(epoch, lr=lr_D, loss=loss_epoch['DL']))

        print('-' * 150)

        net['D'].eval()
        phase = 'val'
        psnrs = []
        ssims = []
        for idx, data in enumerate(test_data_loader):
            im_hazy = data['hazy']
            im_gt = data['clear_img']
            # filename = data['hazyname']
            im_hazy = im_hazy.to(device)
            im_gt = im_gt.to(device)

            with torch.set_grad_enabled(False):
                dehaze = net['D'](im_hazy)
                psnr1 = 10 * torch.log10(1 / F.mse_loss(dehaze, im_gt)).item()
                ssim1 = SSIM(dehaze, im_gt).item()
                ssims.append(ssim1)
                psnrs.append(psnr1)
                if (ii + 1) % 50 == 0:
                    log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}/{:0>3d}, mae={:.2e}, ' + \
                              'psnr={:4.2f}, ssim={:5.4f}'
                    print(log_str.format(epoch + 1, args['epochs'], phase, ii + 1, num_iter_epoch_test,
                                         psnr1, ssim1))
        psnr_per_epoch = np.mean(psnrs)
        ssim_per_epoch = np.mean(ssims)

        print('{:s}: PSNR={:4.2f}, SSIM={:5.4f}'.format(phase, psnr_per_epoch, ssim_per_epoch))
        print('-' * 150)

        # save model
        model_prefix = "{:.4f}_{:.4f}_our_model_".format(psnr_per_epoch, ssim_per_epoch)
        save_path_model = str(Path(args['model_dir']) / (model_prefix + str(epoch + 1) + '.pth'))
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': {x: net[x].state_dict() for x in ['D', 'T', 'P_c', 'A']},
            'optimizer_state_dict': {x: optimizer[x].state_dict() for x in ['D', 'T', "P_c", 'A']},
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
