import argparse
import math
import os
from itertools import combinations

models_dict = {'mse_model':reversed(["mse200", "mse400", "mse800", "mse1600", "mse3200", "mse6400", "mse12800", "mse25600"]),
                'msssim_model':reversed(["msssim4", "msssim8", "msssim16", "msssim32", "msssim64", "msssim128", "msssim320", "msssim640"])}
target_bpp_list = [1.5,1.0,0.5,0.25,0.12,0.06]
def select_rd(rd_log_dir):
    for key_models, value_models in models_dict.items():
        if os.path.exists(os.path.join(rd_log_dir,key_models+'_RD.log')):
            os.remove(os.path.join(rd_log_dir,key_models+'_RD.log'))
        model_rd_lines_list = []
        lambda_list = []
        for value_model in value_models:
            target_log_path = os.path.join(rd_log_dir,value_model+'_RD.log')
            if os.path.exists(target_log_path):
                with open(target_log_path,'r') as f:
                    rd_lines = f.readlines()
                    model_rd_lines_list.append(rd_lines)
                    if key_models == 'mse_model':
                        lamb = str(int(value_model.replace('mse',''))//100)
                    else:
                        lamb = str(float(value_model.replace('msssim', '')) / 100)
                    lambda_list.append(lamb)
                f.close()
        for img_idx in range(96):
            img_rd_lines_list = []
            for i,rd_lines in enumerate(model_rd_lines_list):
                line_list = rd_lines[img_idx].split('\t')
                line_list.insert(1,lambda_list[i])
                img_rd_lines_list.append('\t'.join(line_list))
            img_rd_dict = {}
            real_bpp_list = []
            for img_rd_line in img_rd_lines_list:
                real_bpp = float(img_rd_line.split('\t')[2])
                real_bpp_list.append(real_bpp)
                img_rd_dict[real_bpp] = img_rd_line
            real_bpp_num = len(real_bpp_list)
            tar_bpp_num = len(target_bpp_list)
            if real_bpp_num < tar_bpp_num:
                real_target_bpp_list = target_bpp_list[tar_bpp_num-real_bpp_num:]
            else:
                real_target_bpp_list = target_bpp_list
            real_tar_bpp_num = len(real_target_bpp_list)
            real_bpp_com_list = list(combinations(real_bpp_list, real_tar_bpp_num))
            com_idx = 0
            min_bpp_dev = 100000.0
            for i, real_bpp_com in enumerate(real_bpp_com_list):
                bpp_dev = 0.0
                for bpp_idx in range(len(real_target_bpp_list)):
                    bpp_dev += abs(real_bpp_com[bpp_idx]-real_target_bpp_list[bpp_idx])/real_target_bpp_list[bpp_idx]
                if bpp_dev < min_bpp_dev:
                    min_bpp_dev = bpp_dev
                    com_idx = i
            real_bpp_com = real_bpp_com_list[com_idx]
            rd_lines_write_list = []
            if len(real_bpp_com) < 6:
                rd_lines_write_list=['\n' for i in range(6-len(real_bpp_com))]
            for real_bpp in real_bpp_com:
                rd_lines_write_list.append(img_rd_dict[real_bpp])
            with open(os.path.join(rd_log_dir,key_models+'_RD.log'),'a') as f:
                f.writelines(rd_lines_write_list)
            f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--rd_log_dir", type=str, default='./log_test/NIC-0.1_retrain', help="Directory containing RD logs")
    args = parser.parse_args()
    
    select_rd(args.rd_log_dir)
        
