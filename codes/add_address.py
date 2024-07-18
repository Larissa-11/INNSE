import numpy as np
import math


def find_common_factors(num):
    start = 30
    end = 50
    common_factors = []

    for factor in range(start, end + 1):
        if num % factor == 0:
            common_factors.append(factor)
    if len(common_factors) > 0:
        return common_factors[0]
    else:
        return start


def split_string_to_groups(string, chunk_size):
    num_groups = len(string) // chunk_size
    groups = []
    start = 0
    for i in range(num_groups):
        end = start + chunk_size
        group = string[start:end]
        groups.append(group)
        start = end
    return groups, num_groups


def add_index_to_groups(LR_img_segments, size):
    size_hex = np.char.mod('%x', size)
    length = len(str(size_hex))
    connected_segments = []
    for row in range(len(LR_img_segments)):
        connected_segments.append(connect(row, LR_img_segments[row], length))
    return connected_segments


def connect(index, LR_img_segment, length):
    index_hex = np.char.mod('%x', index)
    dec_index = str(index_hex).zfill(length)
    one_list = dec_index + LR_img_segment
    return one_list


def divide_index(matrix):
    size_hex = np.char.mod('%x', len(matrix))
    length = len(str(size_hex))

    indexs = []
    datas = []

    for row in range(len(matrix)):
        index, data = divide(matrix[row], length)
        indexs.append(index)
        datas.append(data)

    return indexs, datas


def divide(one_list, index_length):
    index = int(''.join(one_list[:index_length]), 16)
    data = one_list[index_length:]

    return index, data


def sort_order(indexes, data_set):
    flag_index = 0
    if max(indexes) > len(indexes):
        while True:
            if flag_index + 1 not in indexes:
                flag_index += 1
                break
            flag_index += 1
    # noinspection PyUnusedLocal
    if flag_index > 0:
        matrix = [[0 for _ in range(len(data_set[0]))] for _ in range(flag_index)]
    else:
        matrix = [[0 for _ in range(len(data_set[0]))] for _ in range(len(indexes))]

    for index in range(len(matrix)):
        if index not in indexes:
            matrix[index] = data_set[indexes.index(index)]
        else:
            matrix[index] = data_set[indexes.index(index)]
    del indexes, data_set
    return matrix


def correct_indexes(indexes):
    corrected_indexes = []
    used_indexes = set()
    for i in range(len(indexes)):
        if indexes[i] in used_indexes or indexes[i] >= len(indexes):
            corrected_indexes.append(None)
        else:
            corrected_indexes.append(indexes[i])
            used_indexes.add(indexes[i])
    next_index = 0
    for i in range(len(corrected_indexes)):
        if corrected_indexes[i] is None:
            while next_index in used_indexes:
                next_index += 1
            corrected_indexes[i] = next_index
            used_indexes.add(next_index)
            next_index += 1
    return corrected_indexes
