import argparse
import math
import os
import struct
import sys
import time
import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# import Util.AE as AE
import AE
import Model.model as model
from Model.context_model import Weighted_Gaussian

from Util.metrics import evaluate


GPU = False
# index - [0-15]
models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
          "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]


@torch.no_grad()
def inference_rd(im_dirs, out_dir, model_dir, model_index, block_width, block_height):
    M, N2 = 192, 128
    if (model_index == 6) or (model_index == 7) or (model_index == 14) or (model_index == 15):
        M, N2 = 256, 192
    image_comp = model.Image_coding(3, M, N2, M, M//2)
    context = Weighted_Gaussian(M)

    ######################### Load Model #########################
    image_comp.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'.pkl'), map_location='cpu'))
    context.load_state_dict(torch.load(
        os.path.join(model_dir, models[model_index] + r'p.pkl'), map_location='cpu'))

    if GPU:
        image_comp.cuda()
        context.cuda()
    ####################compress each image###################
    for im_dir in im_dirs:
        dec_time = 0
        enc_dec_time_start = time.time()
        bin_dir = os.path.join(out_dir,'enc.bin')
        rec_dir = os.path.join(out_dir,'dec.png')
        file_object = open(bin_dir, 'wb')
        ######################### Read Image #########################
        img = Image.open(im_dir)
        ori_img = np.array(img)
        img = ori_img
        H, W, _ = img.shape
        num_pixels = H * W
        C = 3
        Head = struct.pack('2HB', H, W, model_index)
        file_object.write(Head)
        out_img = np.zeros([H, W, C])
        H_offset = 0
        W_offset = 0
        ######################### spliting Image #########################
        Block_Num_in_Width = int(np.ceil(W / block_width))
        Block_Num_in_Height = int(np.ceil(H / block_height))
        img_block_list = []
        for i in range(Block_Num_in_Height):
            for j in range(Block_Num_in_Width):
                img_block_list.append(img[i * block_height:np.minimum((i + 1) * block_height, H),j * block_width:np.minimum((j + 1) * block_width,W),...])
    
        ######################### Padding Image #########################
        Block_Idx = 0
        y_main_q_list = []
        for img in img_block_list:
            block_H = img.shape[0]
            block_W = img.shape[1]

            tile = 64.
            block_H_PAD = int(tile * np.ceil(block_H / tile))
            block_W_PAD = int(tile * np.ceil(block_W / tile))
            im = np.zeros([block_H_PAD, block_W_PAD, 3], dtype='float32')
            im[:block_H, :block_W, :] = img[:, :, :3]/255.0
            im = torch.FloatTensor(im)
            im = im.permute(2, 0, 1).contiguous()
            im = im.view(1, C, block_H_PAD, block_W_PAD)
            if GPU:
                im = im.cuda()
            print('====> Encoding Image:', im_dir, "%dx%d" % (block_H, block_W), 'to', bin_dir, " Block Idx: %d" % (Block_Idx))
            Block_Idx +=1

            with torch.no_grad():
                y_main, y_hyper = image_comp.encoder(im)
                y_main_q = torch.round(y_main)
                y_main_q = torch.Tensor(y_main_q.numpy().astype(np.int))
                
                ####decoding####
                dec_time_start = time.time()
                rec = image_comp.decoder(y_main_q)
                output_ = torch.clamp(rec, min=0., max=1.0)
                out = output_.data[0].cpu().numpy()
                out = out.transpose(1, 2, 0)
                out_img[H_offset : H_offset + block_H, W_offset : W_offset + block_W, :] = out[:block_H, :block_W, :]
                W_offset += block_W
                if W_offset >= W:
                    W_offset = 0
                    H_offset += block_H
                dec_time += (time.time()-dec_time_start)

                # y_hyper_q = torch.round(y_hyper)

                y_hyper_q, xp2 = image_comp.factorized_entropy_func(y_hyper, 2)
                y_hyper_q = torch.Tensor(y_hyper_q.numpy().astype(np.int))

                hyper_dec = image_comp.p(image_comp.hyper_dec(y_hyper_q))

                xp3, params_prob = context(y_main_q, hyper_dec)
                bpp_hyper = (torch.sum(torch.log(xp2)) / (-np.log(2) * num_pixels)).item()
                bpp_main = (torch.sum(torch.log(xp3)) / (-np.log(2) * num_pixels)).item()
                print('bpp_hyper_info:',bpp_hyper,'bpp_main_info:',bpp_main,'bpp_total_info:',bpp_hyper+bpp_main)

            # Main Arith Encode
            Datas = torch.reshape(y_main_q, [-1]).cpu().numpy().astype(np.int).tolist()
            Max_Main = max(Datas)
            Min_Main = min(Datas)
            sample = np.arange(Min_Main, Max_Main+1+1)  # [Min_V - 0.5 , Max_V + 0.5]
            _, c, h, w = y_main_q.shape
            print("Main Channel:", c)
            sample = torch.FloatTensor(np.tile(sample, [1, c, h, w, 1]))

            # 3 gaussian
            prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = [
                torch.chunk(params_prob, 9, dim=1)[i].squeeze(1) for i in range(9)]
            del params_prob
            # keep the weight summation of prob == 1
            probs = torch.stack([prob0, prob1, prob2], dim=-1)
            del prob0, prob1, prob2

            probs = F.softmax(probs, dim=-1)
            # process the scale value to positive non-zero
            scale0 = torch.abs(scale0)
            scale1 = torch.abs(scale1)
            scale2 = torch.abs(scale2)
            scale0[scale0 < 1e-6] = 1e-6
            scale1[scale1 < 1e-6] = 1e-6
            scale2[scale2 < 1e-6] = 1e-6

            m0 = torch.distributions.normal.Normal(mean0, scale0)
            m1 = torch.distributions.normal.Normal(mean1, scale1)
            m2 = torch.distributions.normal.Normal(mean2, scale2)
            lower = torch.zeros(1, c, h, w, Max_Main-Min_Main+2)
            for i in range(sample.shape[4]):
                # print("CDF:", i)
                lower0 = m0.cdf(sample[:, :, :, :, i]-0.5)
                lower1 = m1.cdf(sample[:, :, :, :, i]-0.5)
                lower2 = m2.cdf(sample[:, :, :, :, i]-0.5)
                lower[:, :, :, :, i] = probs[:, :, :, :, 0]*lower0 + \
                    probs[:, :, :, :, 1]*lower1+probs[:, :, :, :, 2]*lower2
            del probs, lower0, lower1, lower2

            precise = 16
            cdf_m = lower.data.cpu().numpy()*((1 << precise) - (Max_Main -
                                                                Min_Main + 1))  # [1, c, h, w ,Max-Min+1]
            cdf_m = cdf_m.astype(np.int32) + sample.numpy().astype(np.int32) - Min_Main
            cdf_main = np.reshape(cdf_m, [len(Datas), -1])

            # Cdf[Datas - Min_V]
            Cdf_lower = list(map(lambda x, y: int(y[x - Min_Main]), Datas, cdf_main))
            # Cdf[Datas + 1 - Min_V]
            Cdf_upper = list(map(lambda x, y: int(
                y[x - Min_Main]), Datas, cdf_main[:, 1:]))
            AE.encode_cdf(Cdf_lower, Cdf_upper, "main.bin")
            FileSizeMain = os.path.getsize("main.bin")
            print("main.bin: %d bytes" % (FileSizeMain))

            # Hyper Arith Encode
            Min_V_HYPER = torch.min(y_hyper_q).cpu().numpy().astype(np.int).tolist()
            Max_V_HYPER = torch.max(y_hyper_q).cpu().numpy().astype(np.int).tolist()
            _, c, h, w = y_hyper_q.shape
            # print("Hyper Channel:", c)
            Datas_hyper = torch.reshape(
                y_hyper_q, [c, -1]).cpu().numpy().astype(np.int).tolist()
            # [Min_V - 0.5 , Max_V + 0.5]
            sample = np.arange(Min_V_HYPER, Max_V_HYPER+1+1)
            sample = np.tile(sample, [c, 1, 1])
            lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(
                torch.FloatTensor(sample) - 0.5, stop_gradient=False))
            cdf_h = lower.data.cpu().numpy()*((1 << precise) - (Max_V_HYPER -
                                                                Min_V_HYPER + 1))  # [N1, 1, Max-Min+1]
            cdf_h = cdf_h.astype(np.int) + sample.astype(np.int) - Min_V_HYPER
            cdf_hyper = np.reshape(np.tile(cdf_h, [len(Datas_hyper[0]), 1, 1, 1]), [
                                   len(Datas_hyper[0]), c, -1])

            # Datas_hyper [256, N], cdf_hyper [256,1,X]
            Cdf_0, Cdf_1 = [], []
            for i in range(c):
                Cdf_0.extend(list(map(lambda x, y: int(
                    y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, :])))   # Cdf[Datas - Min_V]
                Cdf_1.extend(list(map(lambda x, y: int(
                    y[x - Min_V_HYPER]), Datas_hyper[i], cdf_hyper[:, i, 1:])))  # Cdf[Datas + 1 - Min_V]
            AE.encode_cdf(Cdf_0, Cdf_1, "hyper.bin")
            FileSizeHyper = os.path.getsize("hyper.bin")
            print("hyper.bin: %d bytes" % (FileSizeHyper))

            Head_block = struct.pack('2H4h2I', block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER, FileSizeMain, FileSizeHyper)
            file_object.write(Head_block)
            # cat Head_Infor and 2 files together
            # Head = [FileSizeMain,FileSizeHyper,H,W,Min_Main,Max_Main,Min_V_HYPER,Max_V_HYPER,model_index]
            # print("Head Info:",Head)
            with open("main.bin", 'rb') as f:
                bits = f.read()
                file_object.write(bits)
            f.close()
            with open("hyper.bin", 'rb') as f:
                bits = f.read()
                file_object.write(bits)
            f.close()
            del im
        file_object.close()
        with open(bin_dir, "rb") as f:
            bpp = len(f.read())*8./num_pixels
            print('bpp_total_true:',bpp)
        f.close()
        
        out_img = np.round(out_img * 255.0)
        out_img = out_img.astype('uint8')
        img = Image.fromarray(out_img[:H, :W, :])
        img.save(rec_dir)
        [rgb_psnr, rgb_msssim, yuv_psnr,y_msssim]=evaluate(ori_img,out_img)

        class_name = im_dir.split('/')[-2]
        image_name = im_dir.split('/')[-1].replace('.png','')
        enc_dec_time = time.time() - enc_dec_time_start
        enc_time = enc_dec_time - dec_time
        with open(os.path.join(out_dir,models[model_index]+'_RD.log'), "a") as f:
            f.write(class_name+'/'+image_name+'\t'+str(bpp)+'\t'+str(rgb_psnr)+'\t'+str(rgb_msssim)+'\t'+str(-10*np.log10(1-rgb_msssim))+
                           '\t'+str(yuv_psnr)+'\t'+str(y_msssim)+'\t'+str(-10*np.log10(1-y_msssim))+ 
                           '\t'+str(enc_time)+'\t'+str(dec_time)+'\n')
        f.close()
        del out_img
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input Image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output Bin(encode)/Image(decode)")
    parser.add_argument("-m_dir", "--model_dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("-m", "--model", type=int, default=0, help="Model Index [0-15]")
    parser.add_argument("--block_width", type=int, default=2048, help="coding block width")
    parser.add_argument("--block_height", type=int, default=1024, help="coding block height")
    args = parser.parse_args()
  
    test_images = []
    if os.path.isdir(args.input):
      dirs = os.listdir(args.input)
      for dir in dirs:
        path = os.path.join(args.input, dir)
        if os.path.isdir(path):
          test_images += glob.glob(path + '/*.png')
        if os.path.isfile(path):
          test_images.append(path)
    else:
      test_images.append(args.input)
    
    inference_rd(test_images, args.output, args.model_dir, args.model, args.block_width, args.block_height)
        
