import os, argparse
import numpy as np
import torch
import torchvision.utils as vutils
import time
from PIL import Image
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from models.DCPDN import DCPDN
from models.MSBDN import MSBDNNet
from models.AtmLocal import AtmLocal

abs=os.getcwd()+'/'
print(abs)

def tensorShow(tensors,titles=['haze']):
        fig=plt.figure()
        for tensor,tit,i in zip(tensors,titles,range(len(tensors))):
            img = make_grid(tensor)
            npimg = img.numpy()
            ax = fig.add_subplot(221+i)
            ax.imshow(np.transpose(npimg, (1, 2, 0)))
            ax.set_title(tit)
        plt.show()

parser=argparse.ArgumentParser()
parser.add_argument('--task',type=str,default='Fattal',help='its or ots')
parser.add_argument('--test_imgs',type=str, default='./data/test/',help='Test imgs folder')
opt=parser.parse_args()
dataset=opt.task
img_dir=opt.test_imgs
output_dir=f'./results/'
print("pred_dir:",output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir=""
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#
ckp = torch.load(model_dir, map_location='cpu')
netD = torch.nn.DataParallel(MSBDNNet()).cuda()
netD.load_state_dict(ckp['model_state_dict']['D'])
netT = torch.nn.DataParallel(DCPDN()).cuda()
netT.load_state_dict(ckp['model_state_dict']['T'])
netA = torch.nn.DataParallel(AtmLocal(3, 3)).cuda()
netA.load_state_dict(ckp['model_state_dict']['A'])

netD.eval()

for im in os.listdir(img_dir):
    print(f'\r {im}',end='',flush=True)
    tic_start = time.time()
    haze = Image.open(img_dir+im)
    haze = np.array(haze) / 255
    haze1 = torch.from_numpy(haze).float()
    haze1 = haze1.permute(2, 0, 1)
    haze1 = haze1.cuda().unsqueeze(0)

    # Pad the input if not_multiple_of 8
    img_multiple_of = 16
    height, width = haze1.shape[2], haze1.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, (
            (width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    haze1 = F.pad(haze1, (0, padw, 0, padh), 'reflect')
    with torch.no_grad():
        dehaze = netD(haze1)
        dehaze = dehaze.clamp_(0, 1)
        # transmap = netT(haze1)
        # airlight = netA(haze1)

        dehaze = dehaze[:, :, :height, :width]

    tic_end = time.time()
    print(tic_end-tic_start)


    # vutils.save_image(transmap,output_dir+im.split('.')[0]+'_transmape.png')
    vutils.save_image(dehaze, output_dir + im.split('.')[0] + '_dehaze.png')
    # vutils.save_image(airlight, output_dir + im.split('.')[0] + '_airlight.png')


