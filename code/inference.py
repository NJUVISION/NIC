import argparse
import math
import os
import struct
import sys
import time
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# import Util.AE as AE
import AE
import Model.model as model
from Model.context_model import Weighted_Gaussian


GPU = False
# index - [0-15]
models = ["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600",
          "msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"]

@torch.no_grad()
def encode(im_dir, out_dir, model_dir, model_index, block_width, block_height):
    file_object = open(out_dir, 'wb')
    
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
    ######################### Read Image #########################
    img = Image.open(im_dir)
    img = np.array(img)/255.0
    H, W, _ = img.shape
    num_pixels = H * W
    C = 3
    Head = struct.pack('2HB', H, W, model_index)
    file_object.write(Head)
    ######################### spliting Image #########################
    Block_Num_in_Width = int(np.ceil(W / block_width))
    Block_Num_in_Height = int(np.ceil(H / block_height))
    img_block_list = []
    for i in range(Block_Num_in_Height):
        for j in range(Block_Num_in_Width):
            img_block_list.append(img[i * block_height:np.minimum((i + 1) * block_height, H),j * block_width:np.minimum((j + 1) * block_width,W),...])

    ######################### Padding Image #########################
    Block_Idx = 0
    for img in img_block_list:
        block_H = img.shape[0]
        block_W = img.shape[1]

        tile = 64.
        block_H_PAD = int(tile * np.ceil(block_H / tile))
        block_W_PAD = int(tile * np.ceil(block_W / tile))
        im = np.zeros([block_H_PAD, block_W_PAD, 3], dtype='float32')
        im[:block_H, :block_W, :] = img[:, :, :3]
        im = torch.FloatTensor(im)
        im = im.permute(2, 0, 1).contiguous()
        im = im.view(1, C, block_H_PAD, block_W_PAD)
        if GPU:
            im = im.cuda()
        print('====> Encoding Image:', im_dir, "%dx%d" % (block_H, block_W), 'to', out_dir, " Block Idx: %d" % (Block_Idx))
        Block_Idx +=1

        with torch.no_grad():
            y_main, y_hyper = image_comp.encoder(im)
            y_main_q = torch.round(y_main)
            y_main_q = torch.Tensor(y_main_q.numpy().astype(np.int))

            # y_hyper_q = torch.round(y_hyper)

            y_hyper_q, xp2 = image_comp.factorized_entropy_func(y_hyper, 2)
            y_hyper_q = torch.Tensor(y_hyper_q.numpy().astype(np.int))

            hyper_dec = image_comp.p(image_comp.hyper_dec(y_hyper_q))

            xp3, params_prob = context(y_main_q, hyper_dec)

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
        with open("hyper.bin", 'rb') as f:
            bits = f.read()
            file_object.write(bits)




@torch.no_grad()
def decode(bin_dir, rec_dir, model_dir, block_width, block_height):
    ############### retreive head info ###############
    T = time.time()
    file_object = open(bin_dir, 'rb')

    head_len = struct.calcsize('2HB')
    bits = file_object.read(head_len)
    [H, W, model_index] = struct.unpack('2HB', bits)
    # print("File Info:",Head)
    # Split Main & Hyper bins
    C = 3
    out_img = np.zeros([H, W, C])
    H_offset = 0
    W_offset = 0
    Block_Num_in_Width = int(np.ceil(W / block_width))
    Block_Num_in_Height = int(np.ceil(H / block_height))

    c_main = 192
    c_hyper = 128
    
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
        
    for i in range(Block_Num_in_Height):
        for j in range(Block_Num_in_Width):

            Block_head_len = struct.calcsize('2H4h2I')
            bits = file_object.read(Block_head_len)
            [block_H, block_W, Min_Main, Max_Main, Min_V_HYPER, Max_V_HYPER, FileSizeMain, FileSizeHyper] = struct.unpack('2H4h2I', bits)

            precise, tile = 16, 64.

            block_H_PAD = int(tile * np.ceil(block_H / tile))
            block_W_PAD = int(tile * np.ceil(block_W / tile))

            with open("main.bin", 'wb') as f:
                bits = file_object.read(FileSizeMain)
                f.write(bits)
            with open("hyper.bin", 'wb') as f:
                bits = file_object.read(FileSizeHyper)
                f.write(bits)

            ############### Hyper Decoder ###############
            # [Min_V - 0.5 , Max_V + 0.5]
            sample = np.arange(Min_V_HYPER, Max_V_HYPER+1+1)
            sample = np.tile(sample, [c_hyper, 1, 1])
            lower = torch.sigmoid(image_comp.factorized_entropy_func._logits_cumulative(
                torch.FloatTensor(sample) - 0.5, stop_gradient=False))
            cdf_h = lower.data.cpu().numpy()*((1 << precise) - (Max_V_HYPER -
                                                                Min_V_HYPER + 1))  # [N1, 1, Max - Min]
            cdf_h = cdf_h.astype(np.int) + sample.astype(np.int) - Min_V_HYPER
            T2 = time.time()
            AE.init_decoder("hyper.bin", Min_V_HYPER, Max_V_HYPER)
            Recons = []
            for i in range(c_hyper):
                for j in range(int(block_H_PAD * block_W_PAD / 64 / 64)):
                    # print(cdf_h[i,0,:])
                    Recons.append(AE.decode_cdf(cdf_h[i, 0, :].tolist()))
            # reshape Recons to y_hyper_q   [1, c_hyper, H_PAD/64, W_PAD/64]
            y_hyper_q = torch.reshape(torch.Tensor(
                Recons), [1, c_hyper, int(block_H_PAD / 64), int(block_W_PAD / 64)])

            ############### Main Decoder ###############
            hyper_dec = image_comp.p(image_comp.hyper_dec(y_hyper_q))
            h, w = int(block_H_PAD / 16), int(block_W_PAD / 16)
            sample = np.arange(Min_Main, Max_Main+1+1)  # [Min_V - 0.5 , Max_V + 0.5]

            sample = torch.FloatTensor(sample)

            p3d = (5, 5, 5, 5, 5, 5)
            y_main_q = torch.zeros(1, 1, c_main+10, h+10, w+10)  # 8000x4000 -> 500*250
            AE.init_decoder("main.bin", Min_Main, Max_Main)
            hyper = torch.unsqueeze(context.conv3(hyper_dec), dim=1)

            #
            context.conv1.weight.data *= context.conv1.mask

            for i in range(c_main):
                T = time.time()
                for j in range(int(block_H_PAD / 16)):
                    for k in range(int(block_W_PAD / 16)):

                        x1 = F.conv3d(y_main_q[:, :, i:i+12, j:j+12, k:k+12],
                                      weight=context.conv1.weight, bias=context.conv1.bias)  # [1,24,1,1,1]
                        params_prob = context.conv2(
                            torch.cat((x1, hyper[:, :, i:i+2, j:j+2, k:k+2]), dim=1))

                        # 3 gaussian
                        prob0, mean0, scale0, prob1, mean1, scale1, prob2, mean2, scale2 = params_prob[
                            0, :, 0, 0, 0]
                        # keep the weight  summation of prob == 1
                        probs = torch.stack([prob0, prob1, prob2], dim=-1)
                        probs = F.softmax(probs, dim=-1)

                        # process the scale value to positive non-zero
                        scale0 = torch.abs(scale0)
                        scale1 = torch.abs(scale1)
                        scale2 = torch.abs(scale2)
                        scale0[scale0 < 1e-6] = 1e-6
                        scale1[scale1 < 1e-6] = 1e-6
                        scale2[scale2 < 1e-6] = 1e-6
                        # 3 gaussian distributions
                        m0 = torch.distributions.normal.Normal(mean0.view(1, 1).repeat(
                            1, Max_Main-Min_Main+2), scale0.view(1, 1).repeat(1, Max_Main-Min_Main+2))
                        m1 = torch.distributions.normal.Normal(mean1.view(1, 1).repeat(
                            1, Max_Main-Min_Main+2), scale1.view(1, 1).repeat(1, Max_Main-Min_Main+2))
                        m2 = torch.distributions.normal.Normal(mean2.view(1, 1).repeat(
                            1, Max_Main-Min_Main+2), scale2.view(1, 1).repeat(1, Max_Main-Min_Main+2))
                        lower0 = m0.cdf(sample-0.5)
                        lower1 = m1.cdf(sample-0.5)
                        lower2 = m2.cdf(sample-0.5)  # [1,c,h,w,Max-Min+2]

                        lower = probs[0:1]*lower0+probs[1:2]*lower1+probs[2:3]*lower2
                        cdf_m = lower.data.cpu().numpy()*((1 << precise) - (Max_Main -
                                                                            Min_Main + 1))  # [1, c, h, w ,Max-Min+1]
                        cdf_m = cdf_m.astype(np.int) + \
                            sample.numpy().astype(np.int) - Min_Main

                        pixs = AE.decode_cdf(cdf_m[0, :].tolist())
                        y_main_q[0, 0, i+5, j+5, k+5] = pixs

                print("Decoding Channel (%d/192), Time (s): %0.4f" % (i, time.time()-T))
            del hyper, hyper_dec
            y_main_q = y_main_q[0, :, 5:-5, 5:-5, 5:-5]
            rec = image_comp.decoder(y_main_q)

            output_ = torch.clamp(rec, min=0., max=1.0)
            out = output_.data[0].cpu().numpy()
            out = out.transpose(1, 2, 0)
            out_img[H_offset : H_offset + block_H, W_offset : W_offset + block_W, :] = out[:block_H, :block_W, :]
            W_offset += block_W
            if W_offset >= W:
                W_offset = 0
                H_offset += block_H
    out_img = np.round(out_img * 255.0)
    out_img = out_img.astype('uint8')
    img = Image.fromarray(out_img[:H, :W, :])
    img.save(rec_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True, help="Input Image")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output Bin(encode)/Image(decode)")
    parser.add_argument("-m_dir", "--model_dir", type=str, required=True, help="Directory containing trained models")
    parser.add_argument("-m", "--model", type=int, default=0, help="Model Index [0-5]")
    parser.add_argument('--encode', dest='coder_flag', action='store_true')
    parser.add_argument('--decode', dest='coder_flag', action='store_false')
    parser.add_argument("--block_width", type=int, default=2048, help="coding block width")
    parser.add_argument("--block_height", type=int, default=1024, help="coding block height")
    args = parser.parse_args()
  
    T = time.time()
    #if args.coder_flag:
    if True:
        encode(args.input, args.output, args.model_dir, args.model, args.block_width, args.block_height)
    else:
        decode(args.input, args.output, args.model_dir, args.block_width, args.block_height)
    print("Time (s):", time.time() - T)
