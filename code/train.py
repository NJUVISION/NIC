import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

import Model.model as model
import Util.torch_msssim as torch_msssim
from Model.context_model import Weighted_Gaussian
import time

def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 2 every 3 epochs"""
    if epoch < 10:
        lr = init_lr
    else:
        lr = init_lr * (0.5 ** ((epoch-7) // 3))
    if lr < 1e-6:
        lr = 1e-6
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
        
def train(args):
    # NIC_Dataset
    train_data = ImageFolder(root='/data/ljp105/NIC_Dataset/train/', transform=transforms.Compose(
        [transforms.ToTensor()]))
    train_loader = DataLoader(train_data, batch_size=12,
                              shuffle=True, num_workers=8)

    image_comp = model.Image_coding(3, args.M, args.N2, args.M, args.M//2).cuda()
    context = Weighted_Gaussian(args.M).cuda()

    model_existed = os.path.exists(os.path.join(args.out_dir, 'mse'+str(int(args.lmbda*100))+r'.pkl')) and \
        os.path.exists(os.path.join(args.out_dir, 'mse'+str(int(args.lmbda*100))+r'p.pkl'))
    if model_existed:
        image_comp.load_state_dict(torch.load(os.path.join(args.out_dir, 'mse'+str(int(args.lmbda*100))+r'.pkl')))
        context.load_state_dict(torch.load(os.path.join(args.out_dir, 'mse'+str(int(args.lmbda*100))+r'p.pkl')))
        print('resumed the previous model')

    image_comp = nn.DataParallel(image_comp, device_ids=[0, 1])
    context = nn.DataParallel(context, device_ids=[0, 1])

    opt1 = torch.optim.Adam(image_comp.parameters(), lr=args.lr)
    opt2 = torch.optim.Adam(context.parameters(), lr=args.lr)

    lamb = args.lmbda
    # loss_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    msssim_func = torch_msssim.MS_SSIM(max_val=1.).cuda()
    loss_func = nn.MSELoss()

    for epoch in range(20):
        rec_loss_tmp = 0 
        last_time = time.time()
        train_bpp3_tmp = 0
        train_bpp2_tmp = 0
        mse_tmp = 0
        msssim_tmp = 0
        cur_lr = adjust_learning_rate(opt1, epoch, args.lr)
        _ = adjust_learning_rate(opt2, epoch, args.lr)
        for step, batch_x in enumerate(train_loader):
            batch_x = batch_x[0]
            num_pixels = batch_x.size()[0] * \
                batch_x.size()[2] * batch_x.size()[3]
            batch_x = Variable(batch_x).cuda()

            fake, xp1, xp2, xq1, x3 = image_comp(batch_x, 1)
            xp3, _ = context(xq1, x3)
            
            # MS-SSIM
            # dloss = 1.0 - loss_func(fake, batch_x)
            dloss = loss_func(fake, batch_x)
            msssim = msssim_func(fake, batch_x)

            train_bpp1 = torch.sum(torch.log(xp1)) / (-np.log(2) * num_pixels)
            train_bpp2 = torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels)
            train_bpp3 = torch.sum(torch.log(xp3)) / (-np.log(2) * num_pixels)

            l_rec = lamb * dloss + 0.01 * train_bpp3 + 0.01 * train_bpp2

            opt1.zero_grad()
            opt2.zero_grad()
            l_rec.backward()
            
            # gradient clip
            torch.nn.utils.clip_grad_norm_(image_comp.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(context.parameters(), 5)

            opt1.step()
            opt2.step()
            rec_loss_tmp += l_rec.item()
            mse_tmp += dloss.item()
            msssim_tmp += msssim.item()
            train_bpp3_tmp += train_bpp3.item()
            train_bpp2_tmp += train_bpp2.item()
            if step % 100 == 0:
                with open(os.path.join(args.out_dir, 'train_mse'+str(int(args.lmbda*100))+'.log'), 'a') as fd:
                    time_used = time.time()-last_time
                    last_time = time.time()
                    mse = mse_tmp / (step+1)
                    psnr = 10.0 * np.log10(1./mse)
                    msssim_dB = -10*np.log10(1-(msssim_tmp/(step+1)))
                    bpp_total = (train_bpp3_tmp + train_bpp2_tmp) / (step+1)
                    fd.write('ep:%d step:%d time:%.1f lr:%.8f loss:%.6f MSE:%.6f bpp_main:%.4f bpp_hyper:%.4f bpp_total:%.4f psnr:%.2f msssim:%.2f\n'
                             %(epoch, step, time_used, cur_lr, rec_loss_tmp/(step+1), mse, train_bpp3_tmp/(step+1), train_bpp2_tmp/(step+1), bpp_total, psnr, msssim_dB))
                fd.close()
            #print('epoch', epoch, 'step:', step, 'MSE:', dloss.item(), 'entropy1_loss:', train_bpp1.item(),
            #      'entropy2_loss:', train_bpp2.item(), 'entropy3_loss:', train_bpp3.item())
            if (step+1) % 2000 == 0:
                torch.save(context.module.state_dict(),
                           os.path.join(args.out_dir, 'mse'+str(int(args.lmbda*100))+r'p.pkl'))
                torch.save(image_comp.module.state_dict(),
                           os.path.join(args.out_dir, 'mse'+str(int(args.lmbda*100))+r'.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--M", type=int, default=192, help="the value of M")
    parser.add_argument("--N2", type=int, default=128, help="the value of N2")
    parser.add_argument("--lambda", type=float, default=4, dest="lmbda", help="Lambda for rate-distortion tradeoff.")
    parser.add_argument("--lr", type=float, default=5e-5, help="initial learning rate.")
    parser.add_argument('--out_dir', type=str, default='/output/')
    
    args = parser.parse_args()
    print(args)
    train(args)
