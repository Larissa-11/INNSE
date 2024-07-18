import os.path as osp
import logging
import time
from datetime import datetime
import json
import argparse
from collections import OrderedDict
import numpy as np
import options.options as option
import utils.util as util
from data import create_dataset, create_dataloader, data_handle
from models import create_model
from add_address import find_common_factors, split_string_to_groups, add_index_to_groups
from ptflops import get_model_complexity_info
#### options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options YMAL file.')
opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)
save_path = './CodeBook/CodeBook.txt'
util.mkdirs(
    (path for key, path in opt['path'].items()
     if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'], level=logging.INFO,
                  screen=True, tofile=True)
logger = logging.getLogger('base')


def load_dict_from_file(filepath):
    with open(filepath, 'r') as file:
        dictionary = json.load(file)
    return dictionary


combinations = load_dict_from_file(save_path)

#### Create test dataset and dataloader

test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

start_time = datetime.now()
model = create_model(opt)

# deep_model = model.netG
# print(deep_model)
#
# flops,params=get_model_complexity_info(deep_model,(3,678,1020), as_strings=False)
# print(flops,params)

for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    for data in test_loader:
        if opt['model'] == "I_IRN":

            HR_img = model.feed_data_LQ(data)
            img_path = data['LQ_path'][0]
            img_name = osp.splitext(osp.basename(img_path))[0]
            LR_img = model.downscale(HR_img)
            out_dict = OrderedDict()
            out_dict['LR'] = LR_img.detach()[0].float().cpu()

            LR_img = util.tensor2img(out_dict['LR'])  # uint8
            LR_img_one = LR_img.reshape(-1)
            size = LR_img.shape
            hex_size = ''.join(hex(num)[2:].zfill(2) + 'abcdef' for num in size)
            LR_img_hex = np.char.mod('%x', LR_img_one)
            LR_img_two = [str(num).zfill(2) for num in LR_img_hex]
            LR_img_strings = ''.join(LR_img_two)
            factor = find_common_factors(len(LR_img_strings))
            LR_img_segments, size = split_string_to_groups(LR_img_strings, factor)
            LR_img_segments.insert(0, hex_size.rjust(len(LR_img_segments[0]), '0'))
            LR_img_index = add_index_to_groups(LR_img_segments, size)
            ATCG_sequence = []
            for i in range(len(LR_img_index)):
                sequences = ""
                for char in ''.join(LR_img_index[i]):
                    if char in combinations:
                        sequence = ''.join(combinations[char])
                        sequences += sequence
                ATCG_sequence.append(sequences)
            output_path = "coded_image/" + img_name + ".dna"
            data_handle.write_dna_file(output_path, ATCG_sequence)
            encoding_runtime = (datetime.now() - start_time).total_seconds()
            print("encoding_runtime:", encoding_runtime)
        else:
            real_H = model.feed_data_GT(data)
            LR_img = model.downscale(real_H)
            t_step = len(LR_img)
            path = data['GT_path'][0][0]
            path_parts = path.split('/')
            sub_dir_path = '/' + '/'.join(path_parts[3:-1])
            save_dir = dataset_dir + sub_dir_path
            img_dir = osp.join(save_dir)
            util.mkdir(img_dir)
            for i in range(t_step):
                img_path = data['GT_path'][i][0]
                img_name = osp.splitext(osp.basename(img_path))[0]
                out_dict = OrderedDict()
                out_dict['LR'] = LR_img[i].detach()[0].float().cpu()
                LR = util.tensor2img(out_dict['LR'])  # uint8

                LR_img_one = LR.reshape(-1)
                size = LR.shape
                hex_size = ''.join(hex(num)[2:].zfill(2) + 'abcdef' for num in size)
                LR_img_hex = np.char.mod('%x', LR_img_one)
                LR_img_two = [str(num).zfill(2) for num in LR_img_hex]
                LR_img_strings = ''.join(LR_img_two)
                factor = find_common_factors(len(LR_img_strings))
                LR_img_segments, size = split_string_to_groups(LR_img_strings, factor)
                LR_img_segments.insert(0, hex_size.rjust(len(LR_img_segments[0]), '0'))
                LR_img_index = add_index_to_groups(LR_img_segments, size)
                ATCG_sequence = []
                for i in range(len(LR_img_index)):
                    sequences = ""
                    for char in ''.join(LR_img_index[i]):
                        if char in combinations:
                            sequence = ''.join(combinations[char])
                            sequences += sequence
                    ATCG_sequence.append(sequences)
                output_path = "coded_image/" + sub_dir_path
                print(output_path)
                util.mkdir(output_path)
                output_path = osp.join(output_path, img_name + '.dna')
                data_handle.write_dna_file(output_path, ATCG_sequence)
