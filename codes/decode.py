import os.path as osp
import utils.util as util
from data import data_handle
from add_address import divide_index, sort_order, correct_indexes
import numpy as np
import glob
from datetime import datetime
import random
import argparse
from collections import OrderedDict
import options.options as option
from models import create_model
import cv2
import torch
from data.data_handle import load_dict_from_file
from ptflops import get_model_complexity_info
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
start_time = datetime.now()
model = create_model(opt)
# deep_model = model.netG

# flops,params=get_model_complexity_info(deep_model,(3,32,32), as_strings=True)
# print(flops,params)

DNA_dictionary = opt['DNA_dictionary']

combinations = load_dict_from_file(DNA_dictionary)
DNA_paths = opt['DNA_path']
save_path = opt['save_path']
file_paths = glob.glob(DNA_paths + "/*")


def levenshtein_distance(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


for DNA_path in file_paths:

    img_name = osp.splitext(osp.basename(DNA_path))
    dna_sequences = data_handle.read_dna_file(DNA_path)
    LR_img_index = []
    for i in range(len(dna_sequences)):
        LR_img_index.append([])
        for j in range(0, len(dna_sequences[0]), 6):
            f = False
            for key, value in combinations.items():
                if dna_sequences[i][j:j + 6] == value:
                    LR_img_index[i].append(key)
                    f = True
                    break
            if not f:
                min_distance = float('inf')
                min_element = None
                for key, value in combinations.items():
                    distance = levenshtein_distance(dna_sequences[i][j:j + 6], value)
                    if distance < min_distance:
                        min_distance = distance
                        min_element = key
                LR_img_index[i].append(min_element)
    random.shuffle(LR_img_index)
    indexes, data_set = divide_index(LR_img_index)
    indexes = correct_indexes(indexes)
    LR_img_hex = sort_order(indexes, data_set)
    size_list = []
    first_nonzero_index = next((i for i, item in enumerate(LR_img_hex[0]) if item != '0'), None)
    size_list = LR_img_hex[0][first_nonzero_index:] if first_nonzero_index is not None else []
    hex_string = ''.join(size_list)
    hex_groups = [x for x in hex_string.split('abcdef') if x]
    size = [int(group, 16) for group in hex_groups]
    LR_img_hex.pop(0)
    LR_img_merge = []
    LR_img = []
    for sublist in LR_img_hex:
        LR_img_merge.extend(sublist)
    for i in range(0, len(LR_img_merge), 2):
        LR_img.append(int(''.join(LR_img_merge[i:i + 2]), 16))

    LR_img = np.array(LR_img)
    LR_img = LR_img.reshape((size[0], size[1], size[2]))
    LR_img = LR_img.astype(np.uint8)

    LR_img = LR_img.astype(np.float32) / 255.
    LR_img = cv2.cvtColor(LR_img, cv2.COLOR_RGB2BGR)
    img = torch.tensor(LR_img)
    img = img.unsqueeze(0).permute(0, 3, 1, 2)
    if opt['model'] == "I_IRN":
        LR_img = model.feed_data_R2B(img)
        HR_img = model.upscale(LR_img)
        out_dict = OrderedDict()
        out_dict['HR'] = HR_img.detach()[0].float().cpu()
        HR_img = util.tensor2img(out_dict['HR'])  # uint8
        save_img_path = osp.join(save_path, img_name[0] + '.png')
        util.save_img(HR_img, save_img_path)
        decoding_runtime = (datetime.now() - start_time).total_seconds()
        print("decoding_runtime:", decoding_runtime)

    else:
        img = img.unsqueeze(0)
        LR_img = model.feed_data_R2B(img)
        HR_img = model.upscale(LR_img)
        t_step = len(HR_img)
        for i in range(t_step):
            out_dict = OrderedDict()
            out_dict['HR'] = HR_img[i].detach()[0].float().cpu()
            HR = util.tensor2img(out_dict['HR'])  # uint8
            # save images
            save_img_path = osp.join(save_path, img_name[0] + '.png')
            util.save_img(HR, save_img_path)
